# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import numpy as np
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
        
        state_size = 19
        self.action_size = 16
        self.model = PPO(state_size, self.action_size, hidden_size=HIDDEN_SIZE, 
                         actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, 
                         lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS)
        self.state_norm = Normalizer(state_size)
        self.action_mask = [True]*self.action_size
        self.backoff = 0
        self.backoff_log = []
        self.last_waited_slots = 8

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
        n_10, n_20, n_30, n_50, n_100 = [env.nTxOkDelay[i] for i in range(len(env.nTxOkDelay))]

        # get last equivalent backoff
        if n_deferred_access == 0:
            self.equal_backoff = self.backoff + self.last_waited_slots
        self.last_waited_slots = n_waited_slot

        if self.show_log:
            print("At {:6f}s, delta_time={:.3f}ms, n_bss_stas={}, n_obss_stas={}, <10/20/30/50/100ms={}/{}/{}/{}/{}, \n \
                  in_queue/expired/dropped/sending/acked={}/{}/{}/{}/{}, low/high={}/{}, \n \
                  ac={}, cw={}, n_deferred_access={}, n_waited_slot={}, last_equal_backoff={}".format(
                self.sim_time, self.delta_time, n_bss_stas, n_obss_stas, n_10, n_20, n_30, n_50, n_100, 
                n_in_queue, n_expired, n_dropped, n_sending, n_acked, 
                n_low_delay, n_high_delay, ac.name, cw, n_deferred_access, n_waited_slot, self.equal_backoff
            ))
        
        self.choose_action(env)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self, obs):
        # normalize the obs
        t_ = self.delta_time
        nb_, no_ = obs.nBssStas, obs.nObssStas
        total_stas = max(1, obs.nBssStas + obs.nObssStas)
        n10_, n20_, n30_, n50_, n100_ = [obs.nTxOkDelay[i] / total_stas for i in range(len(obs.nTxOkDelay))]
        ni_ = obs.nInQueuePackets
        ne_ = obs.nExpiredPackets
        nd_ = obs.nDroppedPackets
        ns_ = obs.nSendingPackets
        na_ = obs.nAckedPackets
        nl_, nh_ = obs.nLowDelayPackets, obs.nHighDelayPackets
        cw_ = math.log(obs.cw+1, 2) / 10
        da_ = obs.nDeferredAccess / (obs.cw + 1)
        b_ = obs.waitedSlots

        o = [self.equal_backoff, t_, ni_, ne_, nd_, ns_, na_, nl_, nh_, nb_, no_, n10_, n20_, n30_, n50_, n100_, cw_, da_, b_]
        o = self.state_norm.normalize([o])[0]
        next_state = o.tolist() #np.clip(o, -5.0, 5.0).tolist()
        if self.state:
            # caculate reward
            r_packet = nl_ + 0.5*(na_-nl_-nh_) + 0.1*nh_ - 0.1*nd_ - ne_
            r_tx_fail = -1 if (ns_ > 0 and na_ == 0) else 0
            reward = r_tx_fail + r_packet
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state, self.action_mask)

        # get action mask
        scale = (obs.cw + 1) / self.action_size
        valid_size = math.ceil((obs.cw + 1 - obs.waitedSlots) / max(1, scale))
        self.action_mask = [True]*valid_size + [False]*(self.action_size - valid_size)

        # choose action
        self.action = self.model.choose_action(next_state, self.action_mask, self.enable_learning)
        self.action_log.append((valid_size-1, self.action))
        
        self.backoff = min(int(self.action*scale) + random.randint(0, scale-1), 
                           obs.cw - obs.waitedSlots) if scale > 1 else self.action
        self.backoff_log.append((obs.cw - obs.waitedSlots, self.backoff))
        if self.show_log:
            print("New Backoff={} from [0, {}-{}]".format(self.backoff, obs.cw, obs.waitedSlots))
        self.state = next_state

    def reset(self):
        super().reset()
        self.backoff_log = []
        if self.enable_learning:
            self.model.learn()
            self.plot_loss()
            actor_lr = ACTOR_LR * (1-self.episode/N_EPISODES)
            critic_lr = CRITIC_LR * (1-self.episode/N_EPISODES)
            self.model.update_learning_rate(actor_lr, critic_lr)
        self.model.reset()
    
    def save_model(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        prefix = '{}/node_{}_{}_'.format(save_dir, self.node_id, self.agent_type.name)
        self.model.save_model(prefix+'actor.pt', prefix+'critic.pt')

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        y = self.reward_log
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(self.step_time[:len(y)], y, 'g-', linewidth=1)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Reward')
        fig.savefig('{}/node_{}_{}_reward.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close(fig)

        y1, y2 = zip(*self.action_log)
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.scatter(self.step_time[:len(y1)], y1, c='b', marker='+', s=8)
        ax1.scatter(self.step_time[:len(y2)], y2, c='r', marker='.', s=8, alpha=0.5)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Action')
        fig.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name), dpi=250)
        plt.pause(0.1)
        plt.close(fig)

        y1, y2 = zip(*self.backoff_log)
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.scatter(self.step_time[:len(y1)], [0.5 if x == 0 else x for x in y1], c='b', marker='+', s=8)
        ax1.scatter(self.step_time[:len(y2)], [0.5 if x == 0 else x for x in y2], c='r', marker='.', s=8, alpha=0.5)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_yscale('log', base=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Backoff')
        fig.savefig('{}/node_{}_{}_backoff.png'.format(save_dir, self.node_id, self.agent_type.name), dpi=250)
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