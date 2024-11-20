# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import json
from rl_agent import RlAgent
from cca_rl_agent import CcaRlAgent
from channel_access_rl_agent import ChannelAccessRlAgent

class AgentManager:
    def __init__(self):
        self.agent_dict = {}
        self.active_agent = None
    
    def create_agent(self, node_id, memory_id, agent_type):
        if agent_type == "dummy":
            return RlAgent(node_id, memory_id)
        elif agent_type in {"cca", "CCA", "cca sensitivity", "CCA Sensitivity",
                            "cca-sensitivity", "cca_sensitivity",
                            "CCA-Sensitivity", "CCA_Sensitivity"}:
            return CcaRlAgent(node_id, memory_id)
        elif agent_type in {"channel access", "channel-access", "channel_access",
                            "Channel Access", "Channel-Access", "Channel_Access"}:
            return ChannelAccessRlAgent(node_id, memory_id)
        elif agent_type in {"rate control", "rate_control", "rate-control",
                             "Rate Control", "Rate_Control", "Rate-Control"}:
            pass
        elif agent_type in {"aggre", "aggregation", "Aggre", "Aggregation"}:
            pass
        else:
            print("Agent type error.")
            exit(1)
    
    def register_agents(self, filename):
        with open(filename, 'r') as file:
            doc = json.load(file)
            nodes_info = doc["nodes"]

            for node_info in nodes_info:
                node_id = node_info["id"]
                node_type = node_info["type"]
                agents_info = node_info["agents"]

                if node_type in {"AP", "STA"}:
                    print("Node-{} is not an AI node.".format(node_id))
                    continue
                if len(agents_info) == 0:
                    print("Node-{} has no agent.".format(node_id))
                    continue
                
                memory_id = len(self.agent_dict)
                self.agent_dict[node_id] = []
                for agent_info in agents_info:
                    agent_type = agent_info["type"]
                    rl_agent = self.create_agent(node_id, memory_id, agent_type)
                    self.agent_dict[node_id].append(rl_agent)
            
        if len(self.agent_dict) == 0:
            dummy_agent = self.create_agent(0, 0, "dummy")
            self.agent_dict[0] = [dummy_agent]
        
        return len(self.agent_dict) # return the number of memory vectors used
    
    def get_obs(self, msg_interface):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                if agent.is_active(msg_interface):
                    self.active_agent = agent
                    agent.get_obs(msg_interface)
                    return
    
    def set_action(self, msg_interface):
        self.active_agent.set_action(msg_interface)
    
    def reset(self):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.reset()

    def save_models(self, save_dir):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.save_model(save_dir)

    def plot_results(self, save_dir):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.plot_results(save_dir)