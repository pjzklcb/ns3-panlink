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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
from agent_utils import *
from rl_agent import RlAgent
from models.ppo_gru_binary import PPO


# hyperparameters
HIDDEN_DIM = 64
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
GAMMA = 0.98
LMBDA = 0.95
N_EPOCHS = 50
N_EPISODES = 300
EPS = 0.2
        

class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        self_dim = 5
        neighbor_dim = 2
        self.model = PPO(self_dim, neighbor_dim, hidden_dim=HIDDEN_DIM, 
                         actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, 
                         lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS)
        
        self.neighbor_info = {}
        self.mean_backoff = 1/2
        self.last_waited_slots = 0
        self.last_cw = 15
        self.backoff_log = []
        self.debug_mode = True
        self.pk_log = []
        self.Ek_log = []

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
            self.neighbor_info[neighbor_info.node_id] = neighbor_info
            # print("python get {}/{}".format(len(self.neighbor_info), self.n_neighbor))
            self.more_data = env.moreData

    def set_action(self, msg_interface):
        self.choose_action()
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff
        self.neighbor_info = {}
    
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

        self_features = [math.log2(a_cw), a_da, a_nw, l_cd, l_ci]
        self_log = [self.sim_time, self.delta_time, a_cw, a_da, a_nw, self.min_neighbor_backoff, 1, l_ad, l_cd, l_ci]
        
        neighbor_features = {}
        neighbor_log = []
        for n_id, info in self.neighbor_info.items():
            n_role = info.role
            # n_p = info.period
            n_na = info.n_acks
            n_ad = info.txok_delay
            n_it = info.txok_idle_time
            n_rssi = info.rssi
            n_b = info.backoff
            n_cw = info.cw[0]  # cw of AC_BE
            n_tis = info.txok_interval_stats
            neighbor_features[n_id] = [n_ad, n_it]
            neighbor_log.append([n_id, n_b, n_ad, n_it])
            if self.show_log and n_id == 0:
                print(f"\t Neighbor {n_id}: role={n_role}, txok_delay/idle_time={n_ad:.2f}/{n_it:.2f}ms, "
                      f"acks={n_na}, rssi={n_rssi:.2f}dBm, cw={n_cw}, backoff={n_b} \n"
                      f"\t\t txok_stats: interval=[{', '.join(f'{v:.2f}' for v in n_tis)}]ms")

        if a_da == 0:
            self.self_deque = deque([], maxlen=20)
            self.neighbor_deque = {}
            for n_id in neighbor_features.keys():
                self.neighbor_deque[n_id] = deque([], maxlen=20)
        self.self_deque.append(self_features)
        for n_id in neighbor_features.keys():
            self.neighbor_deque[n_id].append(neighbor_features[n_id])
        next_state = (list(self.self_deque), [list(v) for v in self.neighbor_deque.values()])
        next_transition = (self_log, neighbor_log)
        if self.state:
            # caculate reward
            if a_ns:
                backoff_ratio = self.last_waited_slots / self.last_cw
                self.mean_backoff = 0.9*self.mean_backoff + 0.1*backoff_ratio
                if self.mean_backoff >= 0.5:
                    r_backoff = 1 / (self.mean_backoff + 0.5) - 0.5
                    r_tx = 1 if a_na else -1
                else:
                    r_backoff = -1 / (self.mean_backoff + 1)
                    r_tx = 2*self.mean_backoff if a_na else -1
            else:
                r_tx = 0
                r_backoff = 0
            
            reward = r_tx + r_backoff
            self.reward_log.append([reward, r_tx, r_backoff])
            if len(self.state[1]) > 0 and len(neighbor_features) > 0:
                self.model.store_transition(self.state, self.action, reward, next_state)

            if self.debug_mode:
                self.write_to_csv(self.transition + (self.pk_log[-1], self.Ek_log[-1], self.action, self.backoff, reward))

        # choose action
        if a_da == 0:
            self.last_n_waited_slot = -1
        self.next_Ek = (a_cw + 1 + a_nw) / 2
        if len(neighbor_features) == 0:
            p_k = 1
        else:
            # p_k = (a_nw - self.last_n_waited_slot) / (a_cw + 1 - a_nw) if a_nw < a_cw + 1 else 1
            p_k = 1 - self.model.choose_action(next_state)
        self.pk_log.append(p_k)
        self.Ek_log.append(self.next_Ek)
        self.action = 0 if random.random() <= p_k else 1
        # self.action = random.randint(0, 1)
        self.action_log.append(self.action)
        
        self.backoff = self.action * int(self.next_Ek)
        self.backoff_log.append(self.backoff)
        if self.show_log:
            print("New Backoff={}".format(self.backoff))
        self.state = next_state
        self.transition = next_transition
        self.last_waited_slots = self.access_info.n_waited_slot
        self.last_cw = self.access_info.cw
        self.last_min_backoff = self.min_neighbor_backoff

    def reset(self):
        super().reset()
        self.backoff_log = []
        self.pk_log = []
        self.Ek_log = []
        if self.enable_learning:
            self.model.learn()
            self.plot_loss()
            self.model.reset()
            # update learning rate
            actor_lr = ACTOR_LR * (1-self.episode/N_EPISODES)
            critic_lr = CRITIC_LR * (1-self.episode/N_EPISODES)
            self.model.update_learning_rate(actor_lr, critic_lr)
    
    def save_model(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        prefix = '{}/node_{}_{}_'.format(save_dir, self.node_id, self.agent_type.name)
        self.model.save_model(prefix+'actor.pt', prefix+'critic.pt')

    def write_to_csv(self, transition, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        debug_file = os.path.join(save_dir, 'debug.csv')
        if not os.path.exists(debug_file):
            with open(debug_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['sim_time', 'delta_time', 
                                 'cw', 'n_deferred', 'n_waited_slots',
                                 'min_neighbor_backoff',
                                 'node_id', 'backoff', 'txok_delay', 'txok_idle_time',
                                 'pk', 'Ek', 'action', 'new_backoff', 'reward'])

        self_log, neighbor_log, pk, Ek, action, backoff, reward = transition
        with open(debug_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(self_log + [pk, Ek, action, backoff, reward])
            for sta in neighbor_log:
                writer.writerow(['']*(len(self_log) - len(sta)) + sta)

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        y, y_tx, y_backoff = zip(*self.reward_log)
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(self.step_time[:len(y)], y, 'g-', linewidth=1, alpha=0.5)
        ax1.plot(self.step_time[:len(y_tx)], y_tx, 'b-', linewidth=1, alpha=0.5)
        ax1.plot(self.step_time[:len(y_backoff)], y_backoff, 'r-', linewidth=1, alpha=0.5)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Reward')
        fig.savefig('{}/node_{}_{}_reward.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close(fig)

        y = self.action_log
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.scatter(self.step_time[:len(y)], y, c='r', marker='.', s=8, alpha=0.5)
        ax1.scatter(self.step_time[:len(self.pk_log)], self.pk_log, c='b', marker='.', s=8, alpha=0.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Action')
        fig.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name), dpi=250)
        plt.close(fig)

    def plot_loss(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        actor_loss, critic_loss = zip(*self.model.loss_log)
        plt.figure()
        plt.plot(range(len(actor_loss)), actor_loss, 'b-', linewidth=1, label='Actor loss')
        plt.plot(range(len(critic_loss)), critic_loss, 'r-', linewidth=1, label='Critic loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('{}/node_{}_{}_loss.png'.format(self.save_dir, self.node_id, self.agent_type.name))
        plt.close()