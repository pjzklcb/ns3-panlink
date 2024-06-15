import numpy as np
import matplotlib.pyplot as plt
import torch
from models.dqn import DQN

class ChannelAccessRlAgent:
    def __init__(self, n_total, n_ap, n_sta):
        self.state_size = 8
        self.action_size = 2
        self.rl_aglo = DQN(self.state_size, self.action_size)
        self.obs = None
        self.action = None
        self.action_log = []
        self.reward_log = []

        self.n_total = n_total
        self.n_ap = n_ap
        self.n_sta = n_sta
        self.self.state = np.zeros((n_sta+1, n_total+1))
        self.steps_done = 0
    
    def get_obs(self, msg_interface):
        throughput = 0
        for i in range(self.n_total):
            txNode = msg_interface.GetCpp2PyVector()[i].txNode
            # print("processing i {} txNode {}".format(i, txNode))
            for j in range(self.n_sta+1):
                self.self.state[j, txNode] = msg_interface.GetCpp2PyVector()[i].rxPower[j]
            # self.state[:, txNode] = msg_interface.GetCpp2PyVector()[i].rxPower
            if txNode % self.n_ap == 0:  # record mcs in BSS-0
                self.self.state[int(txNode/self.n_ap)][-1] = msg_interface.GetCpp2PyVector()[i].mcs
            if txNode == self.n_ap:     # record delay and tpt of the VR node
                vrDelay = msg_interface.GetCpp2PyVector()[i].holDelay
                vrThroughput = msg_interface.GetCpp2PyVector()[i].throughput
            # Sum all nodes' throughput
            throughput += msg_interface.GetCpp2PyVector()[i].throughput
        
        print("step = {}, VR avg delay = {} ms, VR UL tpt = {} Mbps, total UL tpt = {} Mbps".format(
            self.steps_done, vrDelay, vrThroughput, throughput))
        
        return [vrDelay, vrThroughput, throughput]
    
    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[0].newCcaSensitivity = -82 + self.action
        self.steps_done += 1

    def choose_action(self, obs):
        throughput, vrDelay, vrThroughput = obs
        
        # RL algorithm here, select action
        alpha = 1
        beta = 5
        vr_constrant = 5
        vrtpt_cons = 14.7
        eta = 1

        cur_state = self.state.reshape(1, -1)[0]
        if self.steps_done:
            reward = alpha * throughput + beta * (vr_constrant - vrDelay) + eta * (vrThroughput - vrtpt_cons)
            self.reward_log.append(reward)
            self.rl_aglo.store_transition(prev_state, action, reward, cur_state)
            prev_state = cur_state
        
        action = self.rl_aglo.choose_action(cur_state)

        self.learn()

        return action
    
    def learn(self):
        self.rl_aglo.learn()

    def plot_result(self):
        pass