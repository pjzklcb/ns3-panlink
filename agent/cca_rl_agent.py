# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import matplotlib.pyplot as plt
from rl_agent import AgentType, RlAgent
from models.ddqn import DDQN


class CcaRlAgent(RlAgent):
    def __init__(self, node_id, memory_id):
        super().__init__(node_id, memory_id)
        self.agent_type = AgentType.RL_CCA_SENSITIVITY
        
        state_size = 3
        action_size = (-62) - (-82) + 1
        self.model = DDQN(state_size, action_size)
        self.new_cca = -82
        self.cca_log = []

    def get_obs(self, msg_interface):
        env = msg_interface.GetCpp2PyVector()[self.memory_id]
        self.sim_time = env.simTime
        ac = 0  # AC_BE
        thpt_req = env.thptReq[ac]  # in Mbps
        thpt = env.thpt[ac]          
        delay_req = env.delayReq[ac]    # in ms
        delay = env.delay[ac]

        obs = [ac, thpt_req, thpt, delay_req, delay]
        if self.show_log:
            print("Recv obs: [" + ' '.join('{:.4f}'.format(x) for x in obs) + "]")
        
        self.choose_action(obs)

    def set_action(self, msg_interface):
        msg_interface.GetPy2CppVector()[self.memory_id].ccaSensitivity = self.new_cca

    def choose_action(self, obs):
        ac, thpt_req, thpt, delay_req, delay = obs
        p = thpt/thpt_req if thpt_req else 0
        d = delay/delay_req if thpt_req else 1

        next_state = [ac, p, d]
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

    def save_model(self, save_dir):
        self.model.save_model('{}/node_{}_model_cca.pt'.format(save_dir, self.node_id))

    def plot_results(self, save_dir):
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