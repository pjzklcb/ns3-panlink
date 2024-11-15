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
from models.ddqn import DDQN

class ChannelAccessRlAgent(RlAgent):
    def __init__(self, node_id, memory_id):
        super().__init__(node_id, memory_id)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS
        
        state_size = 15
        self.action_size = 16
        self.model = DDQN(state_size, self.action_size)
        self.backoff = 0
        self.backoff_log = []
        self.action_mask = [True]*self.action_size

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime
        ac = AcIndex(env.ac)
        thpt_req = env.thptReq[ac.value]  # in Mbps
        thpt = env.thpt[ac.value]          
        delay_req = env.delayReq[ac.value]    # in ms
        delay = env.delay[ac.value]
        self_d2lt = env.selfD2lt
        self_channel_util = env.selfChannelUtil
        ack_ratio = env.ackRatio
        cw = env.cw
        n_deferred_access = env.nDeferredAccess
        n_waited_slot = env.waitedSlots
        n_bss_stas = env.nBssStas
        n_obss_stas = env.nObssStas
        bss_d2lt = env.bssD2lt
        bss_channel_util = env.bssChannelUtil
        other_d2lt = env.otherD2lt
        other_channel_util = env.otherChannelUtil

        if self.show_log:
            print("At {:6f}s, n_bss_stas={}, n_obss_stas={}, \n \
                  ac={}, thpt_req/thpt={:.4f}/{:.4f} Mbps, delay_req/delay={:.3f}/{:.3f} ms, \n \
                  self_d2lt={:.3f}ms, bss_d2lt={:.3f}ms, other_d2lt={:.3f}ms, \n \
                  self_channel_util={:.4f}, bss_channel_util={:.4f}, other_channel_util={:.4f}, \n \
                  ack_ratio={:.4f}, cw={}, n_deferred_access={}, n_waited_slot={}".format(
                self.sim_time, n_bss_stas, n_obss_stas, ac.name, thpt_req, thpt, delay_req, delay, 
                self_d2lt, bss_d2lt, other_d2lt, self_channel_util, bss_channel_util, other_channel_util,
                ack_ratio, cw, n_deferred_access, n_waited_slot
            ))
        
        self.choose_action(env)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self, obs):
        # process the obs
        n_b, n_o = obs.nBssStas, obs.nObssStas
        ac = obs.ac
        p = obs.thpt[obs.ac]/obs.thptReq[obs.ac] if obs.thptReq[obs.ac] else 0
        d = obs.delay[obs.ac]/obs.delayReq[obs.ac] if obs.thptReq[obs.ac] else 1
        sum = obs.selfD2lt + obs.bssD2lt + obs.otherD2lt
        sum = sum if sum else 1
        d_s, d_b, d_o = obs.selfD2lt/sum, obs.bssD2lt/sum, obs.otherD2lt/sum
        u_s, u_b, u_o = obs.selfChannelUtil, obs.bssChannelUtil, obs.otherChannelUtil
        r = obs.ackRatio
        cw = obs.cw
        d_a = obs.nDeferredAccess
        b = obs.waitedSlots

        next_state = [n_b, n_o, ac, p, d, d_s, d_b, d_o, u_s, u_b, u_o, r, cw, d_a, b]
        if self.state:
            # caculate reward
            r_thpt = min(0, p-1)
            r_delay = -5 if d == 0 else 1-d
            r_acked = 1 if r > 0 else -1
            reward = r_thpt + r_delay + r_acked
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state, self.action_mask)
            if self.enable_learning:
                self.model.learn()

        # get action mask
        scale = (cw + 1) / self.action_size
        valid_size = math.floor((cw + 1 - b) / scale) if scale > 1 else cw + 1 - b
        self.action_mask = [True]*valid_size + [False]*(self.action_size - valid_size)

        # choose action
        self.action = self.model.choose_action(next_state, self.action_mask)
        self.action_log.append(self.action)
        
        self.backoff = self.action * scale + random.randint(0, scale-1) if scale > 1 else self.action
        self.backoff_log.append((cw - b, self.backoff))
        if self.show_log:
            print("New Backoff={} from [0, {}-{}]".format(self.backoff, cw, b))
        self.state = next_state

    def save_model(self, save_dir):
        self.model.save_model('{}/node_{}_model_channel_access.pt'.format(save_dir, self.node_id))

    def plot_results(self, save_dir):
        y = self.reward_log
        plt.figure()
        plt.plot(range(len(y)), y, 'g-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_reward_channel_access.png'.format(save_dir, self.node_id))
        plt.close()

        y1, y2 = zip(*self.backoff_log)
        plt.figure()
        plt.scatter(range(len(y1)), y1, c='b', marker='+', s=8)
        plt.scatter(range(len(y2)), y2, c='r', marker='o', s=8, alpha=0.5)
        plt.xlabel('Step number')
        plt.ylabel('Backoff')
        plt.savefig('{}/node_{}_action_channel_access.png'.format(save_dir, self.node_id))
        plt.close()

        y = self.model.loss_log
        plt.figure()
        plt.plot(range(len(y)), y, 'b-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Loss')
        plt.savefig('{}/node_{}_loss_channel_access.png'.format(save_dir, self.node_id))
        plt.close()