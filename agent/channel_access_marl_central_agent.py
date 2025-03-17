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
from models.mappo_lagrange import MAPPO


# hyperparameters
HIDDEN_DIM = 64
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.98
LMBDA = 0.95
N_EPOCHS = 50
N_EPISODES = 200
EPS = 0.2
LAMBDA_ = 0.1
ETA_LAMBDA = 5e-3
        

class ChannelAccessMarlCentralAgent(RlAgent):
    def __init__(self, pretrained_model=None):
        super().__init__(-1, -1, True)
        self.agent_type = AgentType.RL_CHANNEL_ACCESS

        agent_feature_dim = 6
        neighbor_feauture_dim = 2
        self.action_dim = 4
        self.model = MAPPO(agent_feature_dim, neighbor_feauture_dim, self.action_dim, 
                           hidden_dim=HIDDEN_DIM, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, 
                           gamma=GAMMA, lmbda=LMBDA, n_epochs=N_EPOCHS, eps=EPS, 
                           lambda_=LAMBDA_, eta_lambda=ETA_LAMBDA)
        
        self.local_agents = {}
        self.transition = {}
        self.last_transition = None
        self.lambda_log = []

        # load pretrained model
        try:
            prefix = os.path.join(os.path.dirname(__file__), 'models')
            self.model.load_model(os.path.join(prefix, pretrained_model[0]), 
                                  os.path.join(prefix, pretrained_model[1]))
            print("Pretrained model loaded successfully.")
        except:
            print("No pretrained model found.")

    def check_if_new_slot(self, sim_time):
        self.delta_time = sim_time - self.sim_time
        if self.sim_time != 0 and self.delta_time > 3e-6:
            self.archive_experience()
            self.step_time.append(sim_time)
        
        self.sim_time = sim_time
    
    def archive_experience(self):
        # construct next global state
        next_agent_features = []
        next_neighbor_features = {}
        for _, trans in self.transition.items():
            next_agent_features.append(trans['agent_features'])
            for neighbor_id, features in trans['agent_neighbor_features'].items():
                if neighbor_id not in self.transition:
                    if neighbor_id not in next_neighbor_features:
                        next_neighbor_features[neighbor_id] = features
                    else:
                        next_neighbor_features[neighbor_id] = [
                            min(x,y) for x, y in zip(next_neighbor_features[neighbor_id], features)]
        next_neighbor_features = list(next_neighbor_features.values())

        if self.last_transition is not None:
            # calculate reward and constraint violation
            reward = 0
            mu = 0.5
            for node_id, trans in self.transition.items():
                if trans['reward'] != 0:
                    if node_id not in self.last_transition:
                        continue # may sent mgt/action frames that not via queue
                    assert ((trans['reward'] == 1 and reward == 0) or 
                            (trans['reward'] == -2 and reward <= 0)), "Mutiple TX fail and success."
                    reward = trans['reward']
                    mu = min(mu, trans['weighted_backoff_ratio'])
            
            # handle absent case
            if reward == 0:
                min_backoff = 1024
                absent_node_mus = []
                for node_id, trans in self.last_transition.items():
                    if node_id in self.transition:
                        continue

                    last_backoff = trans['action'] if trans['action'] < self.action_dim - 1 else trans['CW']
                    backoff_ratio = (trans['n_waited_slot'] + last_backoff) / trans['CW']
                    weighted_backoff_ratio = 0.98*trans['weighted_backoff_ratio'] + 0.02*backoff_ratio
                    # assume the absent node tx
                    if last_backoff < self.delta_time/9e-6:
                        if min_backoff > last_backoff:
                            absent_node_mus = [weighted_backoff_ratio]
                        elif min_backoff == last_backoff:
                            absent_node_mus.append(weighted_backoff_ratio)
                    
                if len(absent_node_mus) == 1:
                    reward = 1
                    mu = min(mu, absent_node_mus[0])

            # calculate constraint violation
            constraint_violation = 50*max(0, 0.5-mu) #if random.random()<0.95 else 0
            
            self.reward_log.append(reward - self.model.lambda_*constraint_violation)

            # fill the transition
            agent_neighbor_features = []
            actions = []
            for _, trans in self.last_transition.items():
                agent_neighbor_features.append(list(trans['agent_neighbor_features'].values()))
                actions.append(trans['action'])

            # store transition
            self.model.store_transition(self.agent_features, agent_neighbor_features, self.neighbor_features, 
                                        actions, reward, constraint_violation, 
                                        next_agent_features, next_neighbor_features)

        # update 
        self.last_transition = self.transition.copy()
        self.agent_features = next_agent_features
        self.neighbor_features = next_neighbor_features
        self.transition = {}

    def collect_local_obs(self, node_id, local_obs):
        assert node_id in self.local_agents, "Local agent not found."
        assert node_id not in self.transition, "Duplicate local obs."
        reward, state, action = local_obs
        self.transition[node_id] = {
            'reward': reward,
            'agent_features': state[0],
            'agent_neighbor_features': state[1],
            'action': action,
            'CW': 2**state[0][0]-1,
            'n_waited_slot': state[0][2],
            'weighted_backoff_ratio': state[0][5]
        }

    def reset(self):
        self.sim_time = 0
        self.step_time = []
        self.transition = {}
        self.last_transition = None
        if len(self.reward_log):
            self.episode += 1
            self.episode_log.append([np.sum(self.reward_log),
                                     np.sum(self.reward_log) / np.sum(np.array(self.reward_log) != 0)])
        self.reward_log = []

        if self.enable_learning:
            # learn
            self.model.learn()
            self.plot_loss()
            self.model.reset()

            # update local models
            for _, agent in self.local_agents.items():
                agent.model.actor.load_state_dict(self.model.actor.state_dict())

            # update learning rate
            actor_lr = ACTOR_LR * (1-self.episode/N_EPISODES)
            critic_lr = CRITIC_LR * (1-self.episode/N_EPISODES)
            self.model.update_learning_rate(actor_lr, critic_lr)
            self.lambda_log.append(self.model.lambda_)
    
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

        y = self.reward_log
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(self.step_time[:len(y)], y, 'g-', linewidth=1, alpha=0.5)
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

        with open(os.path.join(save_dir, 'episode.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.episode_log)
        
        tot_reward, ave_reward = zip(*self.episode_log)
        fig, ax1 = plt.subplots()
        ax1.plot(tot_reward, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        ax2 = ax1.twinx()
        ax2.plot(ave_reward, 'r-', linewidth=1)
        ax2.set_ylabel('Reward per TX')
        fig.savefig('{}/mappo_{}.png'.format(save_dir, self.agent_type.name))
        plt.close()

        plt.figure()
        plt.plot(self.lambda_log, 'b-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Lambda')
        plt.savefig('{}/mappo_{}_lambda.png'.format(save_dir, self.agent_type.name))
        plt.close()