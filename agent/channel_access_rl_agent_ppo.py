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
from agent_utils import AgentType, AcIndex, Normalizer
from rl_agent import RlAgent
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


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True, pretrained_model=None):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS
        
        state_size = 18
        action_size = 16
        self.model = PPO(state_size, action_size, hidden_size=HIDDEN_SIZE, 
                         actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, 
                         lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS)
        self.state_norm = Normalizer(state_size)
        self.backoff = 0
        self.backoff_log = []
        self.last_waited_slots = 8
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

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        ac = AcIndex(env.ac)
        n_in_queue = env.nInQueuePackets
        n_expired = env.nExpiredPackets
        n_dropped = env.nDroppedPackets
        n_sending = env.nSendingPackets
        n_acked = env.nAckedPackets
        n_low_delay = env.nLowDelayPackets
        n_high_delay = env.nHighDelayPackets
        cw = env.cw
        n_deferred_access = env.nDeferredAccess
        n_waited_slot = env.waitedSlots
        n_bss_stas = env.nBssStas
        n_obss_stas = env.nObssStas
        n_5, n_10, n_20, n_30, n_50, n_100 = env.nTxOkDelay
        n_i5, n_i10, n_i20, n_i30, n_i50, n_i100 = env.nTxOkInterval

        # get last equivalent backoff
        if n_deferred_access == 0:
            self.equal_backoff = self.backoff + self.last_waited_slots
        self.last_waited_slots = n_waited_slot

        if self.show_log:
            print("At {:6f}s, delta_time={:.3f}ms, n_bss_stas={}, n_obss_stas={}, \n \
                  delay<5/10/20/30/50/100ms={}/{}/{}/{}/{}/{}, interval<5/10/20/30/50/100ms={}/{}/{}/{}/{}/{}, \n \
                  in_queue/expired/dropped/sending/acked={}/{}/{}/{}/{}, low/high={}/{}, \n \
                  ac={}, cw={}, n_deferred_access={}, n_waited_slot={}, last_equal_backoff={}".format(
                self.sim_time, self.delta_time, n_bss_stas, n_obss_stas, 
                n_5, n_10, n_20, n_30, n_50, n_100, n_i5, n_i10, n_i20, n_i30, n_i50, n_i100,
                n_in_queue, n_expired, n_dropped, n_sending, n_acked, 
                n_low_delay, n_high_delay, ac.name, cw, n_deferred_access, n_waited_slot, self.equal_backoff
            ))
        
        self.choose_action(env)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self, obs):
        # normalize the obs
        t_ = self.delta_time
        # nb_, no_ = obs.nBssStas, obs.nObssStas
        total_stas = max(1, obs.nBssStas + obs.nObssStas)
        n5_, n10_, n20_, n30_, n50_, n100_ = [v/total_stas for v in obs.nTxOkDelay]
        ni5_, ni10_, ni20_, ni30_, ni50_, ni100_ = [v/total_stas for v in obs.nTxOkInterval]
        ni_ = obs.nInQueuePackets
        # ne_ = obs.nExpiredPackets
        # nd_ = obs.nDroppedPackets
        ns_ = obs.nSendingPackets
        na_ = obs.nAckedPackets
        # nl_, nh_ = obs.nLowDelayPackets, obs.nHighDelayPackets
        cw_ = math.log(obs.cw+1, 2) / 10
        da_ = obs.nDeferredAccess
        b_ = obs.waitedSlots

        # o = [self.equal_backoff, t_, ni_, ne_, nd_, ns_, na_, nl_, nh_, nb_, no_, n10_, n20_, n30_, n50_, n100_, cw_, da_, b_]
        o = [self.equal_backoff, t_, ni_, #ns_, na_, nb_, no_, 
             n5_, n10_, n20_, n30_, n50_, n100_, ni5_, ni10_, ni20_, ni30_, ni50_, ni100_,
             cw_, da_, b_]
        if self.debug_mode:
            self.obs_log.append(o.copy())
        o = self.state_norm.normalize([o])[0]
        next_state = o.tolist()
        if self.state:
            # caculate reward
            # r_packet = nl_ + 0.5*(na_-nl_-nh_) + 0.1*nh_ - 0.1*nd_ - ne_
            # r_tx_fail = -1 if (ns_ > 0 and na_ == 0) else 0
            r_tx = 0 if ns_==0 else (na_ if na_ else -ns_)
            reward = r_tx#_fail + r_packet
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state)

        # choose action
        self.action = self.model.choose_action(next_state, enable_learning=self.enable_learning)
        self.action_log.append(self.action)
        
        # self.backoff = self.action
        self.backoff = self.action * 2
        self.backoff_log.append(self.backoff)
        if self.show_log:
            print("New Backoff={}".format(self.backoff))
        self.state = next_state

    def reset(self):
        super().reset()
        self.backoff_log = []
        if self.enable_learning:
            self.model.learn()
            self.plot_loss()
            self.model.reset()
            self.debug_mode = True if self.episode==199 else False
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
            writer.writerow(['equal_backoff', 'delta_time', 'n_in_queue', #'n_expired', 'n_dropped', 
                             #'n_sending', 'n_acked', 'n_low_delay', 'n_high_delay', 'n_bss', 'n_obss', 
                             'n5', 'n10', 'n20', 'n30', 'n50', 'n100', 
                             'ni5', 'ni10', 'ni20', 'ni30', 'ni50', 'ni100',
                             'cw', 'n_deferred', 'n_waited_slots', 'action', 'new_backoff'])
            for i in range(len(self.obs_log)):
                writer.writerow(self.obs_log[i] + [self.action_log[i]] + [self.backoff_log[i]])

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
        plt.pause(0.1)
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