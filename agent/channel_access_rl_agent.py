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
        
        state_size = 16
        self.action_size = 16
        self.model = DDQN(state_size, self.action_size)
        self.backoff = 0
        self.backoff_log = []
        self.action_mask = [True]*self.action_size

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime
        ac = AcIndex(env.ac)
        self_d2lt = env.selfD2lt
        self_channel_util = env.selfChannelUtil
        n_in_queue = env.nInQueuePackets
        n_expired = env.nExpiredPackets
        n_dropped = env.nDroppedPackets
        n_sending = env.nSendingPackets
        n_acked = env.nAckedPackets
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
            print("At {:6f}s, n_bss_stas={}, n_obss_stas={}, in_queue/expired/dropped/sending/acked={}/{}/{}/{}/{}, \n \
                  ac={}, d2lt=[{:.3f}, {:.3f}, {:.3f}]ms, channel_util=[{:.4f}, {:.4f}, {:.4f}], \n \
                  cw={}, n_deferred_access={}, n_waited_slot={}".format(
                self.sim_time, n_bss_stas, n_obss_stas, n_in_queue, n_expired, n_dropped, n_sending, n_acked, 
                ac.name, self_d2lt, bss_d2lt, other_d2lt, self_channel_util, bss_channel_util, 
                other_channel_util, cw, n_deferred_access, n_waited_slot
            ))
        
        self.choose_action(env)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].backoff = self.backoff

    def choose_action(self, obs):
        # normalize the obs
        nb_, no_ = min(1, obs.nBssStas/10), min(1, obs.nObssStas/20)
        ni_ = obs.nInQueuePackets
        ne_ = obs.nExpiredPackets
        nd_ = obs.nDroppedPackets
        ns_ = obs.nSendingPackets
        na_ = obs.nAckedPackets
        sum = obs.selfD2lt + obs.bssD2lt + obs.otherD2lt
        sum = sum if sum else 1
        ds_, db_, do_ = obs.selfD2lt/sum, obs.bssD2lt/sum, obs.otherD2lt/sum
        us_, ub_, uo_ = obs.selfChannelUtil, obs.bssChannelUtil, obs.otherChannelUtil
        cw_ = math.log(obs.cw+1, 2) / 10
        da_ = obs.nDeferredAccess / obs.cw+1
        b_ = obs.waitedSlots / obs.cw

        next_state = [ni_, ne_, nd_, ns_, na_, nb_, no_, ds_, db_, do_, us_, ub_, uo_, cw_, da_, b_]
        if self.state:
            # caculate reward
            r_tx = 1 if na_ else (-1 if ns_ else 0)
            r_packet = -0.1*nd_ - ne_
            reward = r_tx + r_packet
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state, self.action_mask)
            if self.enable_learning:
                self.model.learn()

        # get action mask
        scale = (obs.cw + 1) / self.action_size
        valid_size = math.ceil((obs.cw + 1 - obs.waitedSlots) / scale) if scale > 1 else obs.cw + 1 - obs.waitedSlots
        self.action_mask = [True]*valid_size + [False]*(self.action_size - valid_size)

        # choose action
        self.action = self.model.choose_action(next_state, self.action_mask)
        self.action_log.append(self.action)
        
        if scale > 1:
            self.backoff = min(obs.cw-obs.waitedSlots, int(self.action * scale + random.randint(0, scale-1)))
        else:
            self.backoff = self.action
        self.backoff_log.append((obs.cw - obs.waitedSlots, self.backoff))
        if self.show_log:
            print("New Backoff={} from [0, {}-{}]".format(self.backoff, obs.cw, obs.waitedSlots))
        self.state = next_state

    def reset(self):
        super().reset()
        self.backoff_log = []

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
        plt.scatter(range(len(y2)), [0.5 if x == 0 else x for x in y2], c='r', marker='.', s=8, alpha=0.5)
        plt.yscale('log', base=2)
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