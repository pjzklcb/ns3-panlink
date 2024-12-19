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
import csv
import numpy as np
import torch
from agent_utils import *
from rl_agent import RlAgent
from models.attention import AttentionFeatureExtractor
from models.ppo import PPO


# hyperparameters
HIDDEN_SIZE = 64
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
GAMMA = 0.98
LMBDA = 0.95
N_EPOCHS = 50
N_EPISODES = 500
EPS = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 64
NUM_HEADS = 4


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

        state_size = 6 + LATENT_DIM
        self.action_size = 4
        self.model = PPO(state_size, self.action_size, hidden_size=HIDDEN_SIZE, 
                         actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, 
                         lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS)
        self.neighbor_info = []
        self.backoff = 0
        self.backoff_log = []
        self.feature_extractor = self._load_feature_extractor()
        self.self_norm = MinMaxScaler()
        self.neighbor_norm = MinMaxScaler(neighbor=True)
        self.debug_mode = False
        self.obs_log = []

        # load pretrained model
        try:
            prefix = os.path.join(os.path.dirname(__file__), 'models')
            self.model.load_model(os.path.join(prefix, pretrained_model[0]), 
                                  os.path.join(prefix, pretrained_model[1]))
            print("Pretrained model loaded successfully.")
        except:
            print("No pretrained model found.")
    
    def _load_feature_extractor(self):
        neighbor_dim = 15
        self_dim = 17
        extractor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset/attn_model3.pth')
        feature_extractor = AttentionFeatureExtractor(neighbor_dim, self_dim, LATENT_DIM, num_heads=NUM_HEADS)
        feature_extractor.load_state_dict(torch.load(extractor_path))
        feature_extractor.eval()
        # freeze all params
        for param in feature_extractor.parameters():
            param.requires_grad = False
        return feature_extractor

    def _get_latent_feature(self, neighbor_features, self_features):
        # neighbor_features shape: (N, neighbor_dim)
        # self_features shape: (1, self_dim)
        with torch.no_grad(): 
            neighbor_features = torch.tensor(neighbor_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            self_features = torch.tensor(self_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            latent_features = self.feature_extractor(neighbor_features, self_features)
        
        return latent_features
    
    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        if not self.more_data:
            self.last_nw = 0 if len(self.backoff_log) == 0 else self.access_info.n_waited_slot
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

        self_features = [a_cw, a_da, a_nw, a_nq, a_ns, a_na] + \
                        [l_ns, l_na, l_td, l_ad, l_cu] + \
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

        # extract latent features
        latent_features = [0] * LATENT_DIM
        if len(neighbor_features) != 0:
            # normalize the obs
            normalized_self_features = self.self_norm.normalize([self_features[6:]])
            normalized_neighbor_features = self.neighbor_norm.normalize(neighbor_features)

            # extract latent features
            latent_features = self._get_latent_feature(normalized_neighbor_features, normalized_self_features)
            latent_features = latent_features.squeeze().tolist() # (LATENT_DIM,)

        next_state = self_features[:6] + latent_features
        self.obs_log.append(next_state + [self.n_contention, self.min_neighbor_backoff])
        if self.state:
            # caculate reward
            r_tx = a_na if a_na else -a_ns
            r_wait = 1 if a_da and (a_nw - self.last_nw) == 0 and self.action else 0
            reward = r_tx + r_wait
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state)

        # choose action
        self.action = self.model.choose_action(next_state, enable_learning=self.enable_learning)
        # self.action = random.randint(0, self.access_info.cw - a_nw)
        # self.action = 0 if obs.minNeighborBackoff else 1
        self.action_log.append(self.action)
        
        self.backoff = self.action * 8
        self.backoff_log.append(self.backoff)
        if self.show_log:
            print("New Backoff={}".format(self.backoff))
        self.state = next_state

    def reset(self):
        super().reset()
        self.backoff_log = []
        self.obs_log = []
        if self.enable_learning:
            self.model.learn()
            self.plot_loss()
            self.model.reset()
            self.debug_mode = True if self.episode%50 == 0 else False
            # update learning rate
            actor_lr = ACTOR_LR * (1-self.episode/N_EPISODES)
            critic_lr = CRITIC_LR * (1-self.episode/N_EPISODES)
            self.model.update_learning_rate(actor_lr, critic_lr)
    
    def save_model(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        prefix = '{}/node_{}_{}_'.format(save_dir, self.node_id, self.agent_type.name)
        self.model.save_model(prefix+'actor.pt', prefix+'critic.pt')

    def write_to_csv(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        with open(os.path.join(save_dir, 'debug.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['cw', 'n_deferred', 'n_waited_slots', 'n_in_queue', 'n_sending', 'n_acked'] + \
                            [str(i) for i in range(64)] + \
                            ['nn', 'min_neighbor_backoff', 'action', 'new_backoff', 'reward'])
            for i in range(len(self.obs_log) - 1):
                writer.writerow(self.obs_log[i] + [self.action_log[i]] + [self.backoff_log[i]] + [self.reward_log[i]])

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if self.debug_mode:
            self.write_to_csv(save_dir)

        y = self.reward_log
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(self.step_time[:len(y)], y, 'g-', linewidth=1)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Reward')
        fig.savefig('{}/node_{}_{}_reward.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close(fig)

        y = self.action_log
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.scatter(self.step_time[:len(y)], y, c='r', marker='.', s=8, alpha=0.5)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
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