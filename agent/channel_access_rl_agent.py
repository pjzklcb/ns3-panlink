import numpy as np
import matplotlib.pyplot as plt
from models.dqn import DQN

class ChannelAccessRlAgent:
    def __init__(self, state_size, action_size, n_total, n_ap, n_sta, show_log=False):
        self.state_size = state_size
        self.action_size = action_size
        self.rl_aglo = DQN(state_size, action_size)
        self.state = None
        self.action = None
        self.action_log = []
        self.reward_log = []

        self.n_total = n_total
        self.n_ap = n_ap
        self.n_sta = n_sta
        self.steps_done = 0
        self.show_log = show_log
    
    def get_obs(self, msg_interface):
        throughput, vrDelay, vrThroughput = 0, 0, 0
        for i in range(self.n_total):
            txNode = msg_interface.GetCpp2PyVector()[i].txNode
            if txNode == self.n_ap:     # record delay and tpt of the VR node
                vrDelay = msg_interface.GetCpp2PyVector()[i].holDelay
                vrThroughput = msg_interface.GetCpp2PyVector()[i].throughput
            # Sum all nodes' throughput
            throughput += msg_interface.GetCpp2PyVector()[i].throughput
        
        if self.show_log:
            print("step = {}, VR avg delay = {} ms, VR UL tpt = {} Mbps, total UL tpt = {} Mbps".format(
                self.steps_done, vrDelay, vrThroughput, throughput))
        return [vrDelay, vrThroughput, throughput]
    
    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[0].newCcaSensitivity = -82 + self.action
        if self.show_log:
            print("new CCA: {}".format(-82 + self.action))

    def choose_action(self, msg_interface):
        # RL algorithm here, select action
        throughput, vrDelay, vrThroughput = self.get_obs(msg_interface)
        
        alpha = 1
        beta = 5
        vr_constrant = 5
        vrtpt_cons = 14.7
        eta = 1

        next_state = [throughput, vrDelay, vrThroughput]
        if self.state:
            reward = alpha * throughput + beta * (vr_constrant - vrDelay) + eta * (vrThroughput - vrtpt_cons)
            self.reward_log.append(reward)
            self.rl_aglo.store_transition(self.state, self.action, reward, next_state)
            
        self.action = self.rl_aglo.choose_action(next_state)
        self.action_log.append(self.action)
        
        self.state = next_state
        self.steps_done += 1

        return self.action
    
    def learn(self):
        self.rl_aglo.learn()

    def plot_results(self):
        pass