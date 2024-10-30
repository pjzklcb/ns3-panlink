# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import matplotlib.pyplot as plt
from base_agent import AgentType, BaseAgent
from models.ddqn import DDQN


class CcaAgent(BaseAgent):
    def __init__(self, node_id, memory_id):
        super().__init__(node_id, memory_id)
        self.agent_type = AgentType.RL_CCA_SENSITIVITY
        
        state_size = 4
        action_size = (-62) - (-82) + 1
        self.model = DDQN(state_size, action_size)
        self.new_cca = -82
        self.cca_log = []

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime

        self.choose_action()

    def set_action(self, msg_interface):
        msg_interface.GetCpp2PyVector()[self.memory_id].ccaSensitivity = self.new_cca

    def choose_action(self):
        pass

    def save_model(self, save_dir):
        self.model.save_model('{}/learn/node_{}_model_cca.pt'.format(save_dir, self.node_id))

    def plot_results(self, save_dir):
        y = self.reward_log
        plt.figure()
        plt.plot(range(len(y)), y, 'g-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Reward')
        plt.savefig('{}/learn/node_{}_reward_cca.png'.format(save_dir, self.node_id))
        plt.close()

        y = self.cca_log
        plt.figure()
        plt.plot(range(len(y)), y, 'r-', linewidth=1, alpha=0.5, marker='o', markersize=3)
        plt.xlabel('Step number')
        plt.ylabel('Cca sensitivity')
        plt.savefig('{}/learn/node_{}_action_cca.png'.format(save_dir, self.node_id))
        plt.close()

        y = self.model.loss_log
        plt.figure()
        plt.plot(range(len(y)), y, 'b-', linewidth=1)
        plt.xlabel('Step number')
        plt.ylabel('Loss')
        plt.savefig('{}/learn/node_{}_loss_cca.png'.format(save_dir, self.node_id))
        plt.close()