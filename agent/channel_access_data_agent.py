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
import numpy as np
import torch
from agent_utils import *
from rl_agent import RlAgent
from models.attention import AttentionFeatureExtractor
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
                 chan_util, app_interval_stats, tx_interval_stats, tx_duration_stats):
        self.period = period
        self.n_tx_data = n_tx_data
        self.n_acked_data = n_acked_data
        self.tx_delay = tx_delay
        self.txok_delay = txok_delay
        self.chan_util = chan_util
        self.app_interval_stats = [v for v in app_interval_stats]
        self.tx_interval_stats = [v for v in tx_interval_stats]
        self.tx_duration_stats = [v for v in tx_duration_stats]


class NeighborLinkInfo:
    def __init__(self, role, period, n_tx_data, n_acked_data, tx_delay, txok_delay, 
                 chan_util, rssi, tx_interval_stats, tx_duration_stats):
        self.role = role
        self.period = period
        self.n_tx_data = n_tx_data
        self.n_acked_data = n_acked_data
        self.tx_delay = tx_delay
        self.txok_delay = txok_delay
        self.chan_util = chan_util
        self.rssi = rssi
        self.tx_interval_stats = [v for v in tx_interval_stats]
        self.tx_duration_stats = [v for v in tx_duration_stats]


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        state_size = 23
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
                                            env.waitedSlots, env.nInQueuePackets, 
                                            env.nExpiredPackets, env.nDroppedPackets, 
                                            env.nSendingPackets, env.nAckedPackets, 
                                            env.nLowDelayPackets, env.nHighDelayPackets)
            obs = env.selfLinkObs
            self.link_info = SelfLinkInfo(obs.period, obs.nTxDataPackets, obs.nAckedDataPackets, 
                                          obs.lastTxDataDelay, obs.lastTxOkDelay, obs.channelUtil,
                                          obs.dataArrivalIntervalStats, obs.dataTxIntervalStats,
                                          obs.dataTxDurationStats)
            
        self.n_neighbor = env.nNeighborStas
        if self.n_neighbor > 0:
            obs = env.neighborStaObs
            neighbor_info = NeighborLinkInfo(obs.role, obs.period, obs.nTxDataPackets, 
                                             obs.nAckedDataPackets, obs.lastTxDataDelay, 
                                             obs.lastTxOkDelay, obs.channelUtil, obs.meanRssi,
                                             obs.dataTxIntervalStats, obs.dataTxDurationStats)
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
        l_cu = self.link_info.chan_util
        l_ai = self.link_info.app_interval_stats
        l_tis = self.link_info.tx_interval_stats
        l_tus = self.link_info.tx_duration_stats
        
        if self.show_log:
            print(f"At {self.sim_time:6f}s, delta_time={self.delta_time:.3f}ms, " 
                  f"cw={self.access_info.cw}, n_defer={a_da}, n_waited_slot={a_nw}, "
                  f"in_queue/sending/acked={a_nq}/{a_ns}/{a_na}, \n"
                  f"\t tx_delay/txok_delay={l_td:.2f}/{l_ad:.2f}ms, tx/ack={l_ns}/{l_na}, "
                  f"chan_util={l_cu:.2f}ms, \n"
                  f"\t app_stats: interval=[{', '.join(f'{v:.2f}' for v in l_ai)}]ms, \n"
                  f"\t tx_stats: interval=[{', '.join(f'{v:.2f}' for v in l_tis)}]ms, "
                  f"duration=[{', '.join(f'{v:.2f}' for v in l_tus)}]ms, \n"
                  f"\t n_neighbor={self.n_neighbor}, n_contention={self.n_contention}, "
                  f"min_neighbor_backoff={self.min_neighbor_backoff}")

        self_features = [a_cw, a_da, a_nw, a_nq, a_ns, a_na] + [l_ns, l_na, l_td, l_ad, l_cu] + \
            l_ai + l_tis + l_tus
        
        neighbor_features = []
        for i in range(self.n_neighbor):
            n_role = self.neighbor_info[i].role
            # n_p = self.neighbor_info[i].period
            n_ns = self.neighbor_info[i].n_tx_data
            n_na = self.neighbor_info[i].n_acked_data
            n_td = self.neighbor_info[i].tx_delay
            n_ad = self.neighbor_info[i].txok_delay
            n_cu = self.neighbor_info[i].chan_util
            n_rssi = self.neighbor_info[i].rssi
            n_tis = self.neighbor_info[i].tx_interval_stats
            n_tus = self.neighbor_info[i].tx_duration_stats
            neighbor_features.append([n_role, n_ns, n_na, n_td, n_ad, 
                                      n_cu, n_rssi] + n_tis + n_tus)
            if self.show_log and i == 0:
                print(f"\t Neighbor {i}: role={n_role}, tx_delay/txok_delay={n_td:.2f}/{n_ad:.2f}ms, "
                      f"tx/ack={n_ns}/{n_na}, chan_util={n_cu:.2f}ms, " 
                      f"rssi={n_rssi:.2f}dBm, \n"
                      f"\t\ttx_stats: interval=[{', '.join(f'{v:.2f}' for v in n_tis)}]ms, "
                      f"duration=[{', '.join(f'{v:.2f}' for v in n_tus)}]ms")

        # normalize the obs
        if len(neighbor_features) != 0:
            # save to dataset
            self_feature_names = ['cw', 'n_defer_access', 'n_waited_slots', 'n_in_queue', 
                              'n_sending', 'n_acked', 'n_tx_data', 'n_acked_data', 
                              'last_tx_delay', 'last_txok_delay', 'chan_util', 
                              'app_interval_mean', 'app_interval_std', 'app_interval_min', 
                              'app_interval_max', 'tx_interval_mean', 'tx_interval_std', 
                              'tx_interval_min', 'tx_interval_max', 'tx_duration_mean', 
                              'tx_duration_std', 'tx_duration_min', 'tx_duration_max']
            neighbor_feature_names = ['role', 'n_tx_data', 'n_acked_data', 'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', 'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', 'tx_interval_min', 
                                  'tx_interval_max', 'tx_duration_mean', 'tx_duration_std', 
                                  'tx_duration_min', 'tx_duration_max']
            with h5py.File("contrib/panlink/agent/dataset/data-" + self.timestamp + ".h5", 'a') as f:
                step_group = f.create_group('step_' + str(self.step_count))
                step_group.create_dataset('self_features', data=self_features)
                step_group.create_dataset('neighbor_features', data=neighbor_features)
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