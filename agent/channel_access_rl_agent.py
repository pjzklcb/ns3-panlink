from models.dqn import DQN

class ChannelAccessRlAgent:
    def __init__(self):
        self.n_state = 8
        self.n_action = 2
        self.rl_aglo = DQN(self.n_state, self.n_action)
        self.obs = None
        self.action = None
        self.action_log = []
        self.reward_log = []
    
    def get_obs(self, msg_interface):
        throughput = 0
        for i in range(n_total):
            txNode = msg_interface.GetCpp2PyVector()[i].txNode
            # print("processing i {} txNode {}".format(i, txNode))
            for j in range(n_sta+1):
                state[j, txNode] = msg_interface.GetCpp2PyVector()[i].rxPower[j]
            # state[:, txNode] = msg_interface.GetCpp2PyVector()[i].rxPower
            if txNode % n_ap == 0:  # record mcs in BSS-0
                state[int(txNode/n_ap)][-1] = msg_interface.GetCpp2PyVector()[i].mcs
            if txNode == n_ap:     # record delay and tpt of the VR node
                vrDelay = msg_interface.GetCpp2PyVector()[i].holDelay
                vrThroughput = msg_interface.GetCpp2PyVector()[i].throughput
            # Sum all nodes' throughput
            throughput += msg_interface.GetCpp2PyVector()[i].throughput
    
    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[0].newCcaSensitivity = -82 + self.action