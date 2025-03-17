# Copyright (c) 2025
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent_utils import *
from rl_agent import RlAgent
from models.mappo_lagrange import MAPPO


# hyperparameters
HIDDEN_DIM = 64
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.98
LMBDA = 0.95
N_EPOCHS = 50
EPS = 0.2
LAMBDA_ = 0.2
ETA_LAMBDA = 1e-3
        

class ChannelAccessMarlLocalAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        agent_feature_dim = 6
        neighbor_feauture_dim = 2
        self.action_dim = 4
        self.model = MAPPO(agent_feature_dim, neighbor_feauture_dim, self.action_dim, 
                           hidden_dim=HIDDEN_DIM, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, 
                           gamma=GAMMA, lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS, 
                           lambda_=LAMBDA_, eta_lambda=ETA_LAMBDA)
        
        self.neighbor_info = []
        self.weighted_backoff_ratio = 0.5
        self.backoff_ratio_log = []
        self.mean_backoff_ratio_log = []
        self.debug_mode = True
        self.central_agent = None

        # load pretrained model
        try:
            prefix = os.path.join(os.path.dirname(__file__), 'models')
            self.model.load_model(os.path.join(prefix, pretrained_model[0]), 
                                  os.path.join(prefix, pretrained_model[1]))
            print("Pretrained model loaded successfully.")
        except:
            print("No pretrained model found.")

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
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff
        self.neighbor_info = []
    
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
        # l_ais = self.link_info.app_interval_stats
        l_tis = self.link_info.tx_interval_stats
        
        if self.show_log:
            print(f"At {self.sim_time:6f}s, delta_time={a_dt:.3f}ms, " 
                  f"cw={a_cw}, n_defer={a_da}, n_waited_slot={a_nw}, "
                  f"in_queue/sending/acked={a_nq}/{a_ns}/{a_na}, \n"
                  f"\t txok_delay={l_ad:.2f}ms, tx/ack={l_ns}/{l_na}, "
                  f"tx_stats: interval=[{', '.join(f'{v:.2f}' for v in l_tis)}]ms, \n"
                  f"\t chan_fail_delay/idle_time={l_cd:.2f}/{l_ci:.2f}ms, "
                  f"n_chan_fail={l_cf}, idle_ratio={l_ir*100:.2f}%, \n"
                  f"\t n_neighbor/contention={self.n_neighbor}/{self.n_contention}, "
                  f"min_neighbor_backoff={self.min_neighbor_backoff}")

        # caculate last reward and constraint violation
        reward = 0
        if self.state:
            if a_ns:
                backoff_ratio = (self.last_n_waited_slot+self.backoff) / self.last_cw
                self.backoff_ratio_log.append(backoff_ratio)
                self.weighted_backoff_ratio = 0.98*self.weighted_backoff_ratio + 0.02*backoff_ratio

                reward = 1 if a_na else -2
            
            if self.debug_mode:
                self.write_to_csv(self.transition + (self.action, self.backoff, reward))

        self_features = [math.log2(a_cw+1), a_da, a_nw, l_cd, l_ci, self.weighted_backoff_ratio]
        self_log = [self.sim_time, a_ns, a_na, a_cw, a_da, a_nw, self.n_contention, self.weighted_backoff_ratio,
                    self.node_id, l_ad, l_cd, l_ci]
        
        neighbor_features = {}
        neighbor_log = []
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
            neighbor_features[n_id] = [n_ad, n_it]
            neighbor_log.append([n_id, n_b, n_ad, n_it])
            if self.show_log and i == 0:
                print(f"\t Neighbor {n_id}: role={n_role}, txok_delay/idle_time={n_ad:.2f}/{n_it:.2f}ms, "
                      f"acks={n_na}, rssi={n_rssi:.2f}dBm, cw={n_cw}, backoff={n_b} \n"
                      f"\t\t txok_stats: interval=[{', '.join(f'{v:.2f}' for v in n_tis)}]ms")

        next_state = (self_features, neighbor_features)
        next_transition = (self_log, neighbor_log)

        # choose action
        if len(neighbor_features) == 0:
            self.action = 0
        else:
            self.action = self.model.choose_action(self_features, list(neighbor_features.values()))
        self.action_log.append(self.action)
        
        self.backoff = a_cw if self.action == self.action_dim-1 else self.action
        if self.show_log:
            print("New Backoff={}".format(self.backoff))

        if self.central_agent is not None and len(neighbor_features) > 0:
            self.central_agent.collect_local_obs(self.node_id, (reward, next_state, self.action))

        self.state = next_state
        self.transition = next_transition
        self.last_n_waited_slot = self.access_info.n_waited_slot
        self.last_cw = self.access_info.cw

    def reset(self):
        super().reset()
        if len(self.backoff_ratio_log) > 0:
            self.mean_backoff_ratio_log.append([np.mean(np.array(self.backoff_ratio_log)), 
                                                len(self.backoff_ratio_log)])
        self.weighted_backoff_ratio = 0.5
        self.backoff_ratio_log = []
    
    def save_model(self, save_dir=None):
        pass

    def write_to_csv(self, transition, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if save_dir is None:
            return
        
        debug_file = os.path.join(save_dir, 'debug.csv')
        if not os.path.exists(debug_file):
            with open(debug_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sim_time', 'n_sending', 'n_acked', 
                                 'cw', 'n_deferred', 'n_waited_slots',
                                 'n_contention', 'weighted_backoff_ratio',
                                 'node_id', 'backoff', 'txok_delay', 'txok_idle_time',
                                 'action', 'new_backoff', 'reward'])

        self_log, neighbor_log, action, backoff, reward = transition
        with open(debug_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self_log + [action, backoff, reward])
            for sta in neighbor_log:
                writer.writerow(['']*(len(self_log) - len(sta)) + sta)

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if save_dir is None:
            return

        if len(self.action_log) == 0:
            return 
        
        plt.figure()
        plt.hist(self.action_log, bins=self.action_dim)
        plt.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name), dpi=250)
        plt.close()

    def plot_episodes(self, save_dir=None):
        if len(self.mean_backoff_ratio_log) == 0:
            return
        mean_backoff_ratio, tx_counts = zip(*self.mean_backoff_ratio_log)
        fig, ax1 = plt.subplots()
        ax1.plot(mean_backoff_ratio, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mean Backoff', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(tx_counts, 'r-', linewidth=1)
        ax2.set_ylabel('TX counts', color='r')
        ax2.tick_params('y', colors='r')
        plt.savefig('{}/node_{}_{}_mean_backoff.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close(fig)