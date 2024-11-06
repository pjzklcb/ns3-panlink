# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

from enum import Enum

class AgentType(Enum):
    NO_RL = 0
    RL_CCA_SENSITIVITY = 1
    RL_CHANNEL_ACCESS = 2
    RL_RATE_CONTROL = 3
    RL_AGGREGATION = 4


class RlAgent():
    def __init__(self, node_id, memory_id):
        self.node_id = node_id
        self.memory_id = memory_id
        self.agent_type = AgentType.NO_RL
        self.counter = 0
        self.sim_time = 0
        self.state = None
        self.action = None
        self.action_log = []
        self.reward_log = []

        self.enable_learning = True
        self.show_log = True

    def is_active(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        agent_type = env.agentType
        new_counter = env.counter
        if self.agent_type.value == agent_type and self.counter < new_counter:
            self.counter = new_counter
            print("At {:6f}s, Node-{}: Agent-{} triggers interaction with counter={}".format(
                self.sim_time, self.node_id, self.agent_type.name, self.counter))
            return True
        return False

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime

    def set_action(self, msg_interface):
        pass

    def choose_action(self):
        pass

    def save_model(self, save_dir):
        pass

    def plot_results(self, save_dir):
        pass