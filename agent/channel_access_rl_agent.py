# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from agent_utils import AgentType, AcIndex, Normalizer
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
    def __init__(self, ac, cw, n_deferred_access, n_waited_slot, 
                 n_in_queue, n_expired, n_dropped, n_sending, n_acked, 
                 n_low_delay, n_high_delay):
        self.ac = ac
        self.cw = cw
        self.n_deferred_access = n_deferred_access
        self.n_waited_slot = n_waited_slot
        self.n_in_queue = n_in_queue
        self.n_expired = n_expired
        self.n_dropped = n_dropped
        self.n_sending = n_sending
        self.n_acked = n_acked
        self.n_low_delay = n_low_delay
        self.n_high_delay = n_high_delay


class SelfLinkInfo:
    def __init__(self, period, n_tx_data, n_acked_data, tx_delay, txok_delay, 
                 app_mean, app_std, app_min, app_max, 
                 tx_mean, tx_std, tx_min, tx_max, chan_util):
        self.period = period
        self.n_tx_data = n_tx_data
        self.n_acked_data = n_acked_data
        self.tx_delay = tx_delay
        self.txok_delay = txok_delay
        self.app_stats = [app_mean, app_std, app_min, app_max]
        self.tx_stats = [tx_mean, tx_std, tx_min, tx_max]
        self.chan_util = chan_util


class NeighborLinkInfo:
    def __init__(self, role, period, n_tx_data, n_acked_data, tx_delay, txok_delay, 
                 tx_mean, tx_std, tx_min, tx_max, chan_util, rssi):
        self.role = role
        self.period = period
        self.n_tx_data = n_tx_data
        self.n_acked_data = n_acked_data
        self.tx_delay = tx_delay
        self.txok_delay = txok_delay
        self.tx_stats = [tx_mean, tx_std, tx_min, tx_max]
        self.chan_util = chan_util
        self.rssi = rssi


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS
        
        state_size = 19
        self.action_size = 2
        self.model = DDQN(state_size, self.action_size, hidden_size=HIDDEN_SIZE, 
                         batch_size=BATCH_SIZE, gamma=GAMMA, 
                         epsilon=EPSILON_START if self.enable_learning else 0, 
                         learning_rate=LEARNING_RATE, memory_size=MEMORY_SIZE)
        self.state_norm = Normalizer(state_size)
        self.neighbor_info = []
        self.backoff = 0
        self.backoff_log = []

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
                                            env.waitedSlots, env.nInQueuePackets, 
                                            env.nExpiredPackets, env.nDroppedPackets, 
                                            env.nSendingPackets, env.nAckedPackets, 
                                            env.nLowDelayPackets, env.nHighDelayPackets)
            obs = env.selfLinkObs
            self.link_info = SelfLinkInfo(obs.period, obs.nTxDataPackets, obs.nAckedDataPackets, 
                                            obs.lastTxDataDelay, obs.lastTxOkDelay, 
                                            obs.meanDataArrivalInterval, obs.stdDataArrivalInterval,
                                            obs.minDataArrivalInterval, obs.maxDataArrivalInterval, 
                                            obs.meanTxDataInterval, obs.stdTxDataInterval,
                                            obs.minTxDataInterval, obs.maxTxDataInterval, obs.channelUtil)
            
        self.n_neighbor = env.nNeighborStas
        if self.n_neighbor > 0:
            obs = env.neighborStaObs
            neighbor_info = NeighborLinkInfo(obs.role, obs.period, obs.nTxDataPackets, 
                                             obs.nAckedDataPackets, obs.lastTxDataDelay, 
                                             obs.lastTxOkDelay, obs.meanTxDataInterval, 
                                             obs.stdTxDataInterval, obs.minTxDataInterval, 
                                             obs.maxTxDataInterval, obs.channelUtil, obs.meanRssi)
            self.neighbor_info.append(neighbor_info)
            # print("python get {}/{}".format(len(self.neighbor_info), self.n_neighbor))
            self.more_data = env.moreData

    def set_action(self, msg_interface):
        self.choose_action()
        self.neighbor_info = []
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self):
        a_cw = math.log(self.access_info.cw + 1, 2)
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
        l_td = self.link_info.tx_delay
        l_ad = self.link_info.txok_delay
        l_tstats = self.link_info.tx_stats
        l_astats = self.link_info.app_stats
        l_cu = self.link_info.chan_util
        
        if self.show_log:
            print(f"At {self.sim_time:6f}s, delta_time={self.delta_time:.3f}ms, cw={self.access_info.cw}, "
                  f"n_deferred_access={a_da}, n_waited_slot={a_nw}, \n"
                  f"\t in_queue/sending/acked={a_nq}/{a_ns}/{a_na}, tx/ack={l_ns}/{l_na}, "
                  f"chan_util={l_cu:.2f}ms, \n"
                  f"\t app_stats=[{', '.join(f'{v:.2f}' for v in l_astats)}]ms, "
                  f"tx_stats=[{', '.join(f'{v:.2f}' for v in l_tstats)}]ms, \n"
                  f"\t n_neighbor={self.n_neighbor}, n_contention={self.n_contention}, "
                  f"min_neighbor_backoff={self.min_neighbor_backoff}")

        # normalize the obs
        self_feature = [a_cw, a_da, a_nw, a_nq, a_ns, a_na, l_ns, l_na, l_td, l_ad, l_cu] + l_tstats + l_astats
        neighbor_features = []
        for i in range(self.n_neighbor):
            n_role = self.neighbor_info[i].role
            # n_p = self.neighbor_info[i].period
            n_ns = self.neighbor_info[i].n_tx_data
            n_na = self.neighbor_info[i].n_acked_data
            n_td = self.neighbor_info[i].tx_delay
            n_ad = self.neighbor_info[i].txok_delay
            n_tstats = self.neighbor_info[i].tx_stats
            n_cu = self.neighbor_info[i].chan_util
            n_rssi = self.neighbor_info[i].rssi
            neighbor_features.append([n_role, n_ns, n_na, n_td, n_ad, 
                                      n_cu, n_rssi] + n_tstats)
            if self.show_log and i == 0:
                print(f"\t Neighbor {i}: tx/ack={n_ns}/{n_na}, chan_util={n_cu:.2f}ms, rssi={n_rssi:.2f}dBm, "
                      f"tx_stats=[{', '.join(f'{v:.2f}' for v in n_tstats)}]ms")

        o = self_feature
        o = self.state_norm.normalize([o])[0]
        next_state = o.tolist()
        if self.state:
            # caculate reward
            reward = a_na if a_na else -a_ns
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state)
            if self.enable_learning:
                self.model.learn()

        # choose action
        # self.action = self.model.choose_action(next_state)
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