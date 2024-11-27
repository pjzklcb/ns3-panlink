# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

from agent_utils import AgentType
import matplotlib.pyplot as plt

class RlAgent():
    def __init__(self, node_id, memory_id):
        self.node_id = node_id
        self.memory_id = memory_id
        self.agent_type = AgentType.NO_RL
        self.counter = 0
        self.sim_time = 0
        self.delta_time = 0
        self.state = None
        self.action = None
        self.action_log = []
        self.reward_log = []
        self.episode = 0
        self.episode_log = []

        self.enable_learning = True
        self.show_log = True

    def is_active(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.delta_time = (env.simTime - self.sim_time) * 1e3 # ms
        self.sim_time = env.simTime
        agent_type = env.agentType
        new_counter = env.counter
        if self.agent_type.value == agent_type and self.counter < new_counter:
            self.counter = new_counter
            print("At {:6f}s, Node-{}: Agent-{} triggered with counter={}".format(
                self.sim_time, self.node_id, self.agent_type.name, self.counter))
            return True
        return False

    def reset(self):
        self.counter = 0
        self.sim_time = 0
        self.delta_time = 0
        self.state = None
        self.action = None
        self.action_log = []
        if len(self.reward_log):
            self.episode += 1
            self.episode_log.append(sum(self.reward_log))
            self.reward_log = []

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime

    def set_action(self, msg_interface):
        pass

    def choose_action(self):
        pass

    def save_model(self, save_dir):
        self.model.save_model('{}/node_{}_{}_model.pt'.format(save_dir, self.node_id, self.agent_type.name))

    def plot_results(self, save_dir):
        pass

    def plot_episodes(self, save_dir):
        y = self.episode_log
        plt.figure()
        plt.plot(range(len(y)), y, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_{}.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()