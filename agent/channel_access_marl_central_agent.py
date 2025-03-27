# Copyright (c) 2025
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from agent_utils import *
from rl_agent import RlAgent
from models.mappo_attn import MAPPO


# hyperparameters
HIDDEN_DIM = 64
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.98
LMBDA = 0.95
N_EPISODES = 300
LAMBDA_ = 0.01
LAMBDA_LR = 1e-4
        

class ChannelAccessMarlCentralAgent(RlAgent):
    def __init__(self, pretrained_model=None):
        super().__init__(-1, -1, True)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        agent_feature_dim = 5
        neighbor_feauture_dim = 2
        self.action_dim = 4
        self.model = MAPPO(agent_feature_dim, neighbor_feauture_dim, self.action_dim, 
                           hidden_dim=HIDDEN_DIM, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, 
                           gamma=GAMMA, lmbda=LMBDA, lambda_=LAMBDA_, lambda_lr=LAMBDA_LR)
        
        self.local_agents = {}
        self.transition_log = []
        self.constraint_log = {}
        self.trajectory = []
        self.step_transition = {}
        self.lambda_log = []

        # load pretrained model
        try:
            prefix = os.path.join(os.path.dirname(__file__), 'models')
            self.model.load_model(os.path.join(prefix, pretrained_model[0]), 
                                  os.path.join(prefix, pretrained_model[1]))
            print("Pretrained model loaded successfully.")
        except:
            print("No pretrained model found.")

    def archive_step_transition(self):
        agent_features = []
        agent_neighbor_features = []
        neighbor_features = {}
        actions = []
        rewards = []
        constraint_violations = []
        for _, trans in self.step_transition.items():
            u, vu = trans["state"]
            agent_features.append(u)
            agent_neighbor_features.append(list(vu.values()))
            for neighbor_id, features in vu.items():
                if neighbor_id in self.step_transition:
                    continue
                if neighbor_id not in neighbor_features:
                    neighbor_features[neighbor_id] = features
                else:
                    # if agents' obs of the same neighbor are different, choose the minimum feature
                    neighbor_features[neighbor_id] = [min(x,y) for x,y in zip(neighbor_features[neighbor_id], features)]
            
            actions.append(trans["action"])
            if trans["reward"] != 0:
                rewards.append(trans["reward"])
                constraint_violations.append(max(0, trans["constraint_violation"]))

        r = 0 if len(rewards) == 0 else np.mean(rewards)
        c = 0 if len(constraint_violations) == 0 else np.mean(constraint_violations)
        self.trajectory.append([
            agent_features,
            agent_neighbor_features,
            list(neighbor_features.values()),
            actions, r, c
        ])
        self.reward_log.append(r - self.model.lambda_ * c)

        # save trajectory to csv
        # self.write_to_csv()
        self.step_transition = {}

    def store_trajectory(self):
        # build trajectory
        current_time = self.transition_log[0]["sim_time"]
        for trans in self.transition_log:
            timestamp = trans["sim_time"]
            if timestamp - current_time < 2e-6:
                assert trans["node_id"] not in self.step_transition, "Duplicate node id."
            elif timestamp - current_time > 7e-6:
                self.archive_step_transition()
                self.step_time.append(current_time)
                current_time = timestamp
            else:
                if trans["node_id"] in self.step_transition:
                    self.archive_step_transition()
                    self.step_time.append(current_time)
                    current_time = timestamp

            self.step_transition[trans["node_id"]] = trans

        # store transition
        for i in range(len(self.trajectory)-1):
            u, vu, v, a, r, c = self.trajectory[i]
            u_, v_ = self.trajectory[i+1][0], self.trajectory[i+1][2]
            self.model.store_transition(u, vu, v, a, r, c, u_, v_)

    def learn(self):
        if not self.enable_learning:
            return
        
        # learn
        self.store_trajectory()
        lambda_constraint = 0
        for _, c in self.constraint_log.items():
            lambda_constraint += np.mean(c)

        self.model.learn(lambda_constraint)
        self.plot_loss()
        self.model.reset()

        # update local models
        for _, agent in self.local_agents.items():
            agent.model.actor.load_state_dict(self.model.actor.state_dict())

        # update learning rate
        actor_lr = ACTOR_LR * (1-self.episode/N_EPISODES)
        critic_lr = CRITIC_LR * (1-self.episode/N_EPISODES)
        lambda_lr = LAMBDA_LR * (1-self.episode/N_EPISODES)
        self.model.update_learning_rate(actor_lr, critic_lr, lambda_lr)
        self.lambda_log.append(self.model.lambda_)

    def reset(self):
        self.sim_time = 0
        self.step_time = []

        self.learn()

        self.plot_results()
        if len(self.reward_log):
            self.episode += 1
            self.episode_log.append([np.sum(self.reward_log),
                                     np.sum(self.reward_log) / np.sum(np.array(self.reward_log) != 0)])
        self.reward_log = []
        self.transition_log = []
        self.constraint_log = {}
        self.trajectory = []
        self.step_transition = {}
    
    # def write_to_csv(self, save_dir=None):
    #     save_dir = save_dir if save_dir is not None else self.save_dir
    #     if save_dir is None:
    #         return
        
    #     debug_file = os.path.join(save_dir, 'debug_central.csv')
    #     if not os.path.exists(debug_file):
    #         with open(debug_file, 'w') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(['step', 'sim_time', 'node_id', 'cw', 'n_deferred', 'n_waited_slots', 
    #                              'fail_delay', 'fail_idle', 'weighted_backoff_ratio',
    #                              'action', 'reward', 'constraint'])

    #     with open(debug_file, 'a') as f:
    #         writer = csv.writer(f)
    #         row = 0
    #         for node_id, trans in self.step_transition.items():
    #             u, _, v, a, r, c = trans

    def save_model(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if save_dir is None:
            return
        prefix = '{}/mappo_{}_'.format(save_dir, self.agent_type.name)
        self.model.save_model(prefix+'actor.pt', prefix+'critic.pt')

    def plot_results(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if save_dir is None:
            return

        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(self.step_time[:len(self.reward_log)], self.reward_log, 'g-', linewidth=1, alpha=0.5)
        ax1.scatter(self.step_time[::100], ax1.get_ylim()[1]*np.ones(len(self.step_time[::100])), c='k', marker='|')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Reward')
        fig.savefig('{}/mappo_{}_reward.png'.format(save_dir, self.agent_type.name))
        plt.close(fig)

    def plot_loss(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        if save_dir is None:
            return
        
        actor_loss, critic_loss = zip(*self.model.loss_log)
        plt.figure()
        plt.plot(range(len(actor_loss)), actor_loss, 'b-', linewidth=1, label='Actor loss')
        plt.plot(range(len(critic_loss)), critic_loss, 'r-', linewidth=1, label='Critic loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('{}/mappo_{}_loss.png'.format(self.save_dir, self.agent_type.name))
        plt.close()

    def plot_episodes(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        
        tot_reward, ave_reward = zip(*self.episode_log)
        fig, ax1 = plt.subplots()
        ax1.plot(tot_reward, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(ave_reward, 'r-', linewidth=1)
        ax2.set_ylabel('Reward per TX', color='r')
        ax2.tick_params('y', colors='r')
        fig.savefig('{}/mappo_{}.png'.format(save_dir, self.agent_type.name))
        plt.close()

        plt.figure()
        plt.plot(self.lambda_log, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Lambda')
        plt.savefig('{}/mappo_{}_lambda.png'.format(save_dir, self.agent_type.name))
        plt.close()

        with open(os.path.join(save_dir, 'episode.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(tot_reward)
            writer.writerow(ave_reward)
            writer.writerow(self.lambda_log)