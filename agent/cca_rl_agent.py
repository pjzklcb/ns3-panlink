# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import matplotlib.pyplot as plt
from agent_utils import AgentType, AcIndex
from rl_agent import RlAgent
from models.ddqn import DDQN


# hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 2000
LEARNING_RATE = 1e-4
EPSILON_START = 1.0
HIDDEN_SIZE = 64


class CcaRlAgent(RlAgent):
    def __init__(self, node_id, memory_id, enable_learning=True):
        super().__init__(node_id, memory_id, enable_learning)
        self.agent_type = AgentType.RL_CCA_SENSITIVITY
        
        state_size = 6
        action_size = (-62) - (-82) + 1
        self.model = DDQN(state_size, action_size, hidden_size=HIDDEN_SIZE, 
                         batch_size=BATCH_SIZE, gamma=GAMMA, 
                         epsilon=EPSILON_START if self.enable_learning else 0, 
                         learning_rate=LEARNING_RATE, memory_size=MEMORY_SIZE)
        self.new_cca = -82
        self.cca_log = []

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime
        ac = AcIndex.AC_BE
        thpt_req = env.thptReq[ac.value]  # in Mbps
        thpt = env.thpt[ac.value]          
        delay_req = env.delayReq[ac.value]    # in ms
        delay = env.delay[ac.value]
        n_bss_stas = env.nBssStas
        n_obss_stas = env.nObssStas
        min_bss_rssi = env.bssRssi
        max_obss_rssi = env.obssRssi

        obs = [n_bss_stas, n_obss_stas, thpt_req, thpt, delay_req, delay, min_bss_rssi, max_obss_rssi]
        if self.show_log:
            print("At {:6f}s, n_bss_stas={}, n_obss_stas={}, \n \
                  ac={}, thpt_req/thpt={:.4f}/{:.4f} Mbps, delay_req/delay={:.3f}/{:.3f} ms, \n \
                  min_bss_rssi={:.2f}dBm, max_obss_rssi={:.2f}dBm".format(
                self.sim_time, n_bss_stas, n_obss_stas, ac.name, thpt_req, thpt, delay_req, delay, 
                min_bss_rssi, max_obss_rssi))
        
        self.choose_action(obs)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].ccaSensitivity = self.new_cca

    def choose_action(self, obs):
        n_bss_stas, n_obss_stas, thpt_req, thpt, delay_req, delay, min_bss_rssi, max_obss_rssi = obs
        p = thpt/thpt_req if thpt_req else 0
        d = delay/delay_req if thpt_req else 1
        min_rssi = min_bss_rssi+82 if min_bss_rssi else 0
        max_rssi = max_obss_rssi+82

        next_state = [p, d, n_bss_stas, n_obss_stas, min_rssi, max_rssi]
        if self.state:
            # caculate reward
            r_thpt = min(0, p-1)
            r_delay = -5 if d == 0 else 1-d
            reward = r_thpt + r_delay
            self.reward_log.append(reward)
            self.model.store_transition(self.state, self.action, reward, next_state)
            if self.enable_learning:
                self.model.learn()

        # choose action
        self.action = self.model.choose_action(next_state)
        self.action_log.append(self.action)
        self.new_cca = -82 + self.action
        self.cca_log.append(self.new_cca)
        if self.show_log:
            print("New CCA={}".format(self.new_cca))
        self.state = next_state

    def reset(self):
        super().reset()
        self.cca_log = []

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        
        y = self.reward_log
        plt.figure()
        plt.plot(range(len(y)), y, 'g-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_reward_cca.png'.format(save_dir, self.node_id))
        plt.close()

        y = self.cca_log
        plt.figure()
        plt.plot(range(len(y)), y, 'r-', linewidth=1, alpha=0.5, marker='o', markersize=3)
        plt.xlabel('Step number')
        plt.ylabel('Cca sensitivity')
        plt.savefig('{}/node_{}_action_cca.png'.format(save_dir, self.node_id))
        plt.close()

        y = self.model.loss_log
        plt.figure()
        plt.plot(range(len(y)), y, 'b-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Loss')
        plt.savefig('{}/node_{}_loss_cca.png'.format(save_dir, self.node_id))
        plt.close()