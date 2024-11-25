# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import math
import matplotlib.pyplot as plt
import random
from agent_utils import AgentType, AcIndex
from rl_agent import RlAgent
from models.ppo import PPO


# hyperparameters
HIDDEN_SIZE = 64
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
GAMMA = 0.98
LMBDA = 0.95
N_EPOCHS = 10
EPSILON = 0.2


class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id):
        super().__init__(node_id, memory_id)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS
        
        state_size = 17
        self.action_size = 1024
        self.model = PPO(state_size, self.action_size, hidden_size=HIDDEN_SIZE, 
                         actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=GAMMA, 
                         lmbda=LMBDA, n_epochs=N_EPOCHS, epsilon=EPSILON)
        self.backoff = 0
        self.backoff_log = []
        self.average_backoff = 8
        self.last_waited_slots = 8
        self.alpha = 1
        self.action_mask = [True]*self.action_size

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
        n_txok_10ms = env.nTxOkDelay10ms
        n_txok_50ms = env.nTxOkDelay50ms
        n_txok_100ms = env.nTxOkDelay100ms
        n_txok_500ms = env.nTxOkDelay500ms

        if self.show_log:
            print("At {:6f}s, delta_time={:.3f}ms, n_bss_stas={}, n_obss_stas={}, <10/50/100/500ms={}/{}/{}/{}, \n \
                  in_queue/expired/dropped/sending/acked={}/{}/{}/{}/{}, low/high={}/{}, \n \
                  ac={}, cw={}, n_deferred_access={}, n_waited_slot={}".format(
                self.sim_time, self.delta_time, n_bss_stas, n_obss_stas, n_txok_10ms, n_txok_50ms, n_txok_100ms, 
                n_txok_500ms,n_in_queue, n_expired, n_dropped, n_sending, n_acked, 
                n_low_delay, n_high_delay, ac.name, cw, n_deferred_access, n_waited_slot
            ))
        
        self.choose_action(env)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self, obs):
        # normalize the obs
        t_ = self.delta_time/100
        nb_, no_ = min(1, obs.nBssStas/10), min(1, obs.nObssStas/20)
        total_stas = max(1, obs.nBssStas + obs.nObssStas)
        n10_ = obs.nTxOkDelay10ms / total_stas
        n50_ = obs.nTxOkDelay50ms / total_stas
        n100_ = obs.nTxOkDelay100ms / total_stas
        n500_ = obs.nTxOkDelay500ms / total_stas
        ni_ = obs.nInQueuePackets
        ne_ = obs.nExpiredPackets
        nd_ = obs.nDroppedPackets
        ns_ = obs.nSendingPackets
        na_ = obs.nAckedPackets
        nl_, nh_ = obs.nLowDelayPackets, obs.nHighDelayPackets
        cw_ = math.log(obs.cw+1, 2) / 10
        da_ = obs.nDeferredAccess / obs.cw+1
        b_ = obs.waitedSlots / obs.cw

        # adjust reward weight
        if obs.nDeferredAccess == 0:
            last_backoff = self.backoff + self.last_waited_slots
            self.average_backoff = 0.9*self.average_backoff + 0.1*last_backoff
            if self.average_backoff < 4:
                self.alpha = max(0.1, self.alpha/2)
            elif self.average_backoff > 16:
                self.alpha = min(1, self.alpha*2)
        self.last_waited_slots = obs.waitedSlots

        next_state = [t_, ni_, ne_, nd_, ns_, na_, nl_, nh_, nb_, no_, n10_, n50_, n100_, n500_, cw_, da_, b_]
        if self.state:
            # caculate reward
            r_tx = 1 if na_ else (-1 if ns_ else 0)
            r_packet = nl_ + 0.5*(na_-nl_-nh_) + 0.1*nh_ - 0.1*nd_ - ne_
            reward = r_tx + self.alpha*r_packet
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state, self.action_mask)

        # get action mask
        valid_size = obs.cw + 1 - obs.waitedSlots
        self.action_mask = [True]*valid_size + [False]*(self.action_size - valid_size)

        # choose action
        self.action = self.model.choose_action(next_state, self.action_mask)
        self.action_log.append(self.action)
        
        self.backoff = self.action
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
        self.model.reset()
    
    def save_model(self, save_dir):
        self.model.save_model('{}/node_{}_{}_'.format(save_dir, self.node_id, self.agent_type.name))
    
    def load_model(self, load_dir):
        self.model.save_model('{}/node_{}_{}_'.format(load_dir, self.node_id, self.agent_type.name))

    def plot_results(self, save_dir):
        self.save_dir = save_dir
        y = self.reward_log
        plt.figure()
        plt.plot(range(len(y)), y, 'g-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_{}_reward.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()

        y1, y2 = zip(*self.backoff_log)
        plt.figure()
        plt.scatter(range(len(y1)), y1, c='b', marker='+', s=8)
        plt.scatter(range(len(y2)), [0.5 if x == 0 else x for x in y2], c='r', marker='.', s=8, alpha=0.5)
        plt.yscale('log', base=2)
        plt.xlabel('Step number')
        plt.ylabel('Backoff')
        plt.savefig('{}/node_{}_{}_action.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()

    def plot_loss(self):
        actor_loss, critic_loss = zip(*self.model.loss_log)
        plt.figure()
        plt.plot(range(len(actor_loss)), actor_loss, 'b-', linewidth=1, label='Actor loss')
        plt.plot(range(len(critic_loss)), critic_loss, 'r-', linewidth=1, label='Critic loss')
        plt.legend()
        plt.xlabel('Step number')
        plt.ylabel('Loss')
        plt.savefig('{}/node_{}_{}_loss.png'.format(self.save_dir, self.node_id, self.agent_type.name))
        plt.close()