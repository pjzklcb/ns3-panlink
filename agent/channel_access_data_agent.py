# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import h5py
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent_utils import *
from rl_agent import RlAgent
from models.ddqn import DDQN


# hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 2000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
HIDDEN_SIZE = 64


class AccessInfo:
    def __init__(self, ac, cw, n_deferred_access, n_waited_slot, obs):
        self.ac = ac
        self.cw = cw
        self.n_deferred_access = n_deferred_access
        self.n_waited_slot = n_waited_slot
        self.n_in_queue = obs.nInQueuePackets
        self.n_expired = obs.nExpiredPackets
        self.n_dropped = obs.nDroppedPackets
        self.n_sending = obs.nSendingPackets
        self.n_acked = obs.nAckedPackets
        self.n_low_delay = obs.nLowDelayPackets
        self.n_high_delay = obs.nHighDelayPackets


class SelfLinkInfo:
    def __init__(self, obs):
        self.period = obs.period
        self.n_tx_data = obs.nTxDataPackets
        self.n_acked_data = obs.nAckedDataPackets
        self.txok_delay = obs.lastTxOkDelay
        self.n_rx_fail = obs.nRxFail
        self.idle_ratio = obs.idleRatio
        self.app_interval_stats = [v for v in obs.trafficIntervalStats]
        self.tx_interval_stats = [v for v in obs.txDataIntervalStats]


class NeighborLinkInfo:
    def __init__(self, obs):
        self.role = obs.role
        self.period = obs.period
        self.n_acks = obs.nAcks
        self.txok_delay = obs.lastTxOkDelay
        self.rssi = obs.meanRssi
        self.txok_interval_stats = [v for v in obs.txOkIntervalStats]
        self.backoff = obs.backoffSlots
        self.cw = [v for v in obs.cw]


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        state_size = 20
        self.action_size = 4
        self.model = DDQN(state_size, self.action_size, hidden_size=HIDDEN_SIZE, 
                         batch_size=BATCH_SIZE, gamma=GAMMA, 
                         epsilon=EPSILON_START if self.enable_learning else 0, 
                         learning_rate=LEARNING_RATE, memory_size=MEMORY_SIZE)
        
        self.neighbor_info = []
        self.backoff = 0
        self.backoff_log = []
        self.step_count = 0
        self.timestamp = str(int(time.time()))

        # load pretrained model
        try:
            prefix = os.path.join(os.path.dirname(__file__), 'models')
            self.model.load_model(os.path.join(prefix, pretrained_model))
            print("Pretrained model loaded successfully.")
        except:
            print("No pretrained model found.")
        
        if not self.enable_learning:
            self.model.epsilon = 0

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        if not self.more_data:
            self.n_contention = env.nStasWithPackets
            self.min_neighbor_backoff = env.minNeighborBackoff
            self.access_info = AccessInfo(AcIndex(env.ac), env.cw, env.nDeferredAccess, 
                                          env.waitedSlots, env.lastAccessObs)
            self.link_info = SelfLinkInfo(env.selfLinkObs)
            
        self.n_neighbor = env.nNeighborStas
        if self.n_neighbor > 0:
            neighbor_info = NeighborLinkInfo(env.neighborStaObs)
            self.neighbor_info.append(neighbor_info)
            # print("python get {}/{}".format(len(self.neighbor_info), self.n_neighbor))
            self.more_data = env.moreData

    def set_action(self, msg_interface):
        self.choose_action()
        self.neighbor_info = []
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff
    
    def choose_action(self):
        a_dt = self.delta_time
        a_cw = self.access_info.cw
        a_da = self.access_info.n_deferred_access
        a_nw = self.access_info.n_waited_slot
        a_nq = self.access_info.n_in_queue
        # a_ne = self.access_info.n_expired
        # a_nd = self.access_info.n_dropped
        a_ns = self.access_info.n_sending
        a_na = self.access_info.n_acked
        # a_nl = self.access_info.n_low_delay
        # a_nh = self.access_info.n_high_delay

        # l_p = self.link_info.period
        l_ns = self.link_info.n_tx_data
        l_na = self.link_info.n_acked_data
        l_ad = self.link_info.txok_delay
        l_rf = self.link_info.n_rx_fail
        l_ir = self.link_info.idle_ratio
        l_ais = self.link_info.app_interval_stats
        l_tis = self.link_info.tx_interval_stats
        
        if self.show_log:
            print(f"At {self.sim_time:6f}s, delta_time={self.delta_time:.3f}ms, " 
                  f"cw={self.access_info.cw}, n_defer={a_da}, n_waited_slot={a_nw}, "
                  f"in_queue/sending/acked={a_nq}/{a_ns}/{a_na}, \n"
                  f"\t txok_delay={l_ad:.2f}ms, tx/ack={l_ns}/{l_na}, \n"
                  f"\t app_stats: interval=[{', '.join(f'{v:.2f}' for v in l_ais)}]ms, \n"
                  f"\t tx_stats: interval=[{', '.join(f'{v:.2f}' for v in l_tis)}]ms, \n"
                  f"\t  n_rx_fail={l_rf}, idle_ratio={l_ir*100:.2f}%,  n_neighbor={self.n_neighbor}, "
                  f"n_contention={self.n_contention}, min_neighbor_backoff={self.min_neighbor_backoff}")

        self_features = (
            [a_dt, a_cw, a_da, a_nw, a_nq, a_ns, a_na] + [l_ns, l_na, l_ad, l_rf, l_ir] + 
            l_ais + l_tis
        )
        
        neighbor_features = []
        for i in range(self.n_neighbor):
            n_role = self.neighbor_info[i].role
            # n_p = self.neighbor_info[i].period
            n_na = self.neighbor_info[i].n_acks
            n_ad = self.neighbor_info[i].txok_delay
            n_rssi = self.neighbor_info[i].rssi
            n_b = self.neighbor_info[i].backoff
            n_cw = self.neighbor_info[i].cw[0]  # cw of AC_BE
            n_tis = self.neighbor_info[i].txok_interval_stats
            neighbor_features.append([n_role, n_na, n_ad, n_rssi, n_b, n_cw] + n_tis)
            if self.show_log and i == 0:
                print(f"\t Neighbor {i}: role={n_role}, txok_delay={n_ad:.2f}ms, "
                      f"acks={n_na}, rssi={n_rssi:.2f}dBm, cw={n_cw}, backoff={n_b} \n"
                      f"\t\t txok_stats: interval=[{', '.join(f'{v:.2f}' for v in n_tis)}]ms")

        # save to dataset
        if len(neighbor_features) != 0:
            # save to dataset
            self_feature_names = ['delta_time', 'cw', 'n_defer_access', 'n_waited_slots', 'n_in_queue', 
                                'n_sending', 'n_acked', 'n_tx_data', 'n_acked_data', 
                                'last_tx_delay', 'last_txok_delay', 'n_rx_fail', 'idle_ratio',
                                'app_interval_mean', 'app_interval_std', 'app_interval_min', 
                                'app_interval_max', 'tx_interval_mean', 'tx_interval_std', 
                                'tx_interval_min', 'tx_interval_max']
            neighbor_feature_names = ['role', 'n_acks', 'last_txok_delay', 'mean_rssi', 
                                      'backoff', 'cw',
                                      'tx_interval_mean', 'tx_interval_std', 'tx_interval_min', 
                                      'tx_interval_max']
            with h5py.File("contrib/panlink/agent/dataset/data-" + self.timestamp + ".h5", 'a') as f:
                step_group = f.create_group('step_' + str(self.step_count))
                step_group.create_dataset('self_features', data=self_features)
                step_group.create_dataset('neighbor_features', data=neighbor_features)
                step_group.create_dataset('min_neighbor_backoff', data=self.min_neighbor_backoff)
                step_group['self_features'].attrs['feature_names'] = self_feature_names
                step_group['neighbor_features'].attrs['feature_names'] = neighbor_feature_names
            self.step_count += 1

        next_state = self_features
        if self.state:
            # caculate reward
            reward = a_na if a_na else -a_ns
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state)
            if self.enable_learning:
                self.model.learn()

        # choose action
        # self.action = self.model.choose_action(next_state) * 8
        self.action = random.randint(0, self.access_info.cw - a_nw)
        self.action_log.append(self.action)
        
        self.backoff = self.action
        self.backoff_log.append(self.backoff)
        if self.show_log:
            print("New Backoff={} from [0, {}-{}]".format(self.backoff, self.access_info.cw, a_nw))
        self.state = next_state

    def reset(self):
        super().reset()
        self.backoff_log = []

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        y = self.reward_log
        plt.figure()
        plt.plot(range(len(y)), y, 'g-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_{}_reward.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()

        y = self.action_log
        plt.figure()
        plt.scatter(range(len(y)), [0.5 if x == 0 else x for x in y], c='r', marker='.', s=8, alpha=0.5)
        plt.yscale('log', base=2)
        plt.xlabel('Step number')
        plt.ylabel('Action')
        plt.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()

        y = self.model.loss_log
        plt.figure()
        plt.plot(range(len(y)), y, 'b-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Loss')
        plt.savefig('{}/node_{}_{}_loss.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()