# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from agent_utils import AgentType

class RlAgent():
    def __init__(self, node_id, memory_id, enable_learning=True):
        self.node_id = node_id
        self.memory_id = memory_id
        self.agent_type = AgentType.NO_RL
        self.counter = 0
        self.more_data = False
        self.sim_time = 0
        self.delta_time = 0
        self.state = None
        self.action = None
        self.action_log = []
        self.reward_log = []
        self.step_time = []     # (sim_time, counter)
        self.episode = 0
        self.episode_log = []

        self.enable_learning = enable_learning
        self.show_log = False
        self.save_dir = '.'

    def is_active(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.delta_time = (env.simTime - self.sim_time) * 1e3 # ms
        self.sim_time = env.simTime
        agent_type = env.agentType
        new_counter = env.counter
        if self.agent_type.value == agent_type and self.counter < new_counter:
            self.counter = new_counter
            if self.show_log:
                print("At {:6f}s, Node-{}: Agent-{} triggered with counter={}".format(
                    self.sim_time, self.node_id, self.agent_type.name, self.counter))
            self.step_time.append(self.sim_time)

            return True
        return False

    def reset(self):
        self.counter = 0
        self.more_data = False
        self.sim_time = 0
        self.delta_time = 0
        self.state = None
        self.action = None
        self.action_log = []
        self.step_time = []
        if len(self.reward_log):
            self.episode += 1
            self.episode_log.append(np.sum(self.reward_log))
        self.reward_log = []

    def get_obs(self, msg_interface):
        pass
    
    def has_more_data(self):
        return self.more_data
    
    def set_action(self, msg_interface):
        pass

    def choose_action(self):
        pass

    def save_model(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if hasattr(self, 'model'):
            self.model.save_model('{}/node_{}_{}_model.pt'.format(save_dir, self.node_id, self.agent_type.name))

    def plot_results(self, save_dir=None):
        pass

    def plot_episodes(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir

        with open(os.path.join(save_dir, 'episode.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.episode_log)
            
        plt.figure()
        plt.plot(self.episode_log, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('{}/node_{}_{}.png'.format(save_dir, self.node_id, self.agent_type.name))
        plt.close()