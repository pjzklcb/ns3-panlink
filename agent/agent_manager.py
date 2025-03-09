# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import json
from rl_agent import RlAgent
from cca_rl_agent import CcaRlAgent
# from channel_access_rl_agent_manual import ChannelAccessRlAgent
from channel_access_rl_agent_ppo_lagrange import ChannelAccessRlAgent
from channel_access_marl_central_agent import ChannelAccessMarlCentralAgent
from channel_access_marl_local_agent import ChannelAccessMarlLocalAgent

class AgentManager:
    def __init__(self, marl=False):
        self.marl = marl
        self.agent_dict = {}
        self.memory_id = {}
        self.active_agent = None
    
    def create_agent(self, node_id, agent_info):
        memory_id = self.memory_id[node_id]
        agent_type = agent_info["type"]
        enable_learning = agent_info["enable_learning"] if "enable_learning" in agent_info else True
        pretrained_model = agent_info["pretrained_model"] if "pretrained_model" in agent_info else None
        
        if self.marl:
            print("Node-{} installing {}_local_agent".format(node_id, agent_type))
            if agent_type in {"channel_access"}:
                local_agent = ChannelAccessMarlLocalAgent(node_id, memory_id, enable_learning, pretrained_model)
                tmp_agent = ChannelAccessMarlCentralAgent(pretrained_model)
            
            if enable_learning:
                if 'central' not in self.agent_dict:
                    self.agent_dict['central'] = []

                central_agent = None
                for agent in self.agent_dict['central']:
                    if agent.agent_type == local_agent.agent_type:
                        central_agent = agent
                        break
                if central_agent is None:
                    central_agent = tmp_agent
                    self.agent_dict['central'].append(central_agent)
                
                local_agent.central_agent = central_agent
                central_agent.local_agents[node_id] = local_agent
            
            return local_agent
        
        print("Node-{} installing {}_rl_agent".format(node_id, agent_type))
        if agent_type == "dummy":
            return RlAgent(node_id, memory_id, enable_learning)
        elif agent_type in {"cca", "Cca"}:
            return CcaRlAgent(node_id, memory_id, enable_learning)
        elif agent_type in {"channel_access", "Channel_Access"}:
            return ChannelAccessRlAgent(node_id, memory_id, enable_learning, pretrained_model)
        elif agent_type in {"rate_control", "Rate_Control"}:
            pass
        elif agent_type in {"aggregation", "Aggregation"}:
            pass
        else:
            ValueError("Agent type error.")
    
    def register_agents(self, filename):
        with open(filename, 'r') as file:
            doc = json.load(file)
            nodes_info = doc["nodes"]

            for node_info in nodes_info:
                node_id = node_info["id"]
                node_type = node_info["type"]
                agents_info = node_info["agents"]

                if node_type in {"ap", "sta"}:
                    print("Node-{} is not an AI node.".format(node_id))
                    continue
                if len(agents_info) == 0:
                    print("Node-{} has no agent.".format(node_id))
                    continue
                
                if node_id not in self.memory_id:
                    self.memory_id[node_id] = len(self.memory_id)
                
                self.agent_dict[node_id] = []
                for agent_info in agents_info:
                    rl_agent = self.create_agent(node_id, agent_info)
                    self.agent_dict[node_id].append(rl_agent)
            
        if len(self.agent_dict) == 0:
            self.memory_id[0] = 0
            dummy_agent = self.create_agent(0, {"type": "dummy"})
            self.agent_dict[0] = [dummy_agent]
        
        return len(self.agent_dict) # return the number of memory vectors used
    
    def get_active_agent(self, msg_interface):
        for node, agents in self.agent_dict.items():
            if node == 'central':
                continue
            for agent in agents:
                if agent.is_active(msg_interface):
                    self.active_agent = agent
                    return
                
    def get_obs(self, msg_interface):
        self.active_agent.get_obs(msg_interface)
    
    def has_more_data(self):
        return self.active_agent.has_more_data()
  
    def set_action(self, msg_interface):
        self.active_agent.set_action(msg_interface)
    
    def set_save_dir(self, save_dir):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.save_dir = save_dir

    def reset(self):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.reset()

    def save_models(self, save_dir=None):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.save_model(save_dir)

    def plot_results(self, save_dir=None):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.plot_results(save_dir)
    
    def plot_episodes(self, save_dir=None):
        for node, agents in self.agent_dict.items():
            for agent in agents:
                agent.plot_episodes(save_dir)