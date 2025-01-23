# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import csv
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent_utils import *
from rl_agent import RlAgent


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS
        
        self.neighbor_info = []
        self.backoff = 0
        self.mean_backoff = 8
        self.backoff_log = []
        self.debug_mode = True
    
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
        l_cd = self.link_info.chan_fail_delay
        l_ci = self.link_info.chan_fail_idle_time
        l_cf = self.link_info.n_chan_fail
        l_ir = self.link_info.idle_ratio
        l_ais = self.link_info.app_interval_stats
        l_tis = self.link_info.tx_interval_stats
        
        if self.show_log:
            print(f"At {self.sim_time:6f}s, delta_time={a_dt:.3f}ms, " 
                  f"cw={a_cw}, n_defer={a_da}, n_waited_slot={a_nw}, "
                  f"in_queue/sending/acked={a_nq}/{a_ns}/{a_na}, \n"
                  f"\t txok_delay={l_ad:.2f}ms, tx/ack={l_ns}/{l_na}, \n"
                  f"\t app_stats: interval=[{', '.join(f'{v:.2f}' for v in l_ais)}]ms, "
                  f"tx_stats: interval=[{', '.join(f'{v:.2f}' for v in l_tis)}]ms, \n"
                  f"\t chan_fail_delay/idle_time={l_cd:.2f}/{l_ci:.2f}ms, "
                  f"n_chan_fail={l_cf}, idle_ratio={l_ir*100:.2f}%, \n"
                  f"\t n_neighbor/contention={self.n_neighbor}/{self.n_contention}, "
                  f"min_neighbor_backoff={self.min_neighbor_backoff}")
        
        neighbor_features = []
        for i in range(self.n_neighbor):
            n_id = self.neighbor_info[i].node_id
            n_role = self.neighbor_info[i].role
            # n_p = self.neighbor_info[i].period
            n_na = self.neighbor_info[i].n_acks
            n_ad = self.neighbor_info[i].txok_delay
            n_it = self.neighbor_info[i].txok_idle_time
            n_rssi = self.neighbor_info[i].rssi
            n_b = self.neighbor_info[i].backoff
            n_cw = self.neighbor_info[i].cw[0]  # cw of AC_BE
            n_tis = self.neighbor_info[i].txok_interval_stats
            neighbor_features.append([n_id, n_b])
            if self.show_log and i == 0:
                print(f"\t Neighbor {n_id}: role={n_role}, txok_delay/idle_time={n_ad:.2f}/{n_it:.2f}ms, "
                      f"acks={n_na}, rssi={n_rssi:.2f}dBm, cw={n_cw}, backoff={n_b} \n"
                      f"\t\t txok_stats: interval=[{', '.join(f'{v:.2f}' for v in n_tis)}]ms")

        self_features = [self.sim_time, self.delta_time] + [a_cw, a_da, a_nw] + [self.min_neighbor_backoff]
        next_obs = (self_features, neighbor_features)
        next_state = self_features
        if self.state:
            reward = a_na if a_na else -a_ns
            self.reward_log.append(reward)

            if self.debug_mode:
                self.write_to_csv(self.obs + (self.action, self.backoff, reward))

        # choose action
        if a_da == 0:
            self.last_n_waited_slot = -1
            self.next_Ek = (a_cw+1) / 2
        p_k = (a_nw - self.last_n_waited_slot) / (a_cw + 1 - a_nw) if a_nw < a_cw + 1 else 1
        self.next_Ek = (self.next_Ek - p_k * a_nw) / (1 - p_k) if p_k < 1 else a_nw
        next_bk = self.next_Ek - a_nw
        # if self.min_neighbor_backoff == 0:
        #     self.action = 1
        # elif self.min_neighbor_backoff == int(next_bk):
        #     self.action = 0
        # else:
        self.action = 0 if random.random() <= p_k else 1
        # self.action = random.randint(0, 1)
        # self.action = 0 if self.min_neighbor_backoff else 1
        self.action_log.append(self.action)
        
        self.backoff = self.action * int(next_bk)
        self.backoff_log.append(self.backoff)
        if self.show_log:
            print("New Backoff={}".format(self.backoff))
        self.state = next_state
        self.last_n_waited_slot = a_nw
        self.obs = next_obs

    def reset(self):
        super().reset()
        self.backoff_log = []
        self.self_feature_log = []
        self.neighbor_features_log = []

    def write_to_csv(self, transition, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        debug_file = os.path.join(save_dir, 'debug.csv')
        if not os.path.exists(debug_file):
            with open(debug_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sim_time', 'delta_time', 'cw', 'n_deferred', 'n_waited_slots', 'min_neighbor_backoff',
                                 'node_id', 'backoff',
                                 'action', 'new_backoff', 'reward'])

        s_feature, n_features, action, backoff, reward = transition
        with open(debug_file, 'a') as f:
            writer = csv.writer(f)
            if len(n_features) == 0:
                writer.writerow(s_feature + ['', '', action, backoff, reward])
            else:
                writer.writerow(s_feature + n_features[0] + [action, backoff, reward])
                for i in range(1, len(n_features)):
                    writer.writerow(['']*len(s_feature) + n_features[i])

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        y = self.action_log
        plt.figure()
        plt.scatter(range(len(y)), [0.5 if x == 0 else x for x in y], c='r', marker='.', s=8, alpha=0.5)
        plt.yscale('log', base=2)
        plt.xlabel('Step number')
        plt.ylabel('Action')
        plt.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()