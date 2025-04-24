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
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
GAMMA = 0.98
LMBDA = 0.95
LAMBDA_ = 0.01
LAMBDA_LR = 3e-3
CONSTRAINT_VIOLATION_TH = 0
        

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

        # log for plot
        self.lambda_log = []
        self.n_agents_log = []
        self.ave_backoff_ratio_log = []
        self.loss_log = []
        self.entropy_log = []

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
        max_backoff_ratio = 0
        for _, trans in self.step_transition.items():
            u, vu = trans["state"]
            cw, n_waited_slots = 2**u[0]-1, u[2]
            backoff_ratio = n_waited_slots/cw
            agent_features.append(u)
            max_backoff_ratio = max(max_backoff_ratio, backoff_ratio)
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
                rewards.append((backoff_ratio, trans["reward"]))
                constraint_violations.append(trans["constraint_violation"])

        # calculate reward
        reward = 0
        for backoff_ratio, r in rewards:
            if r < 0:
                reward = r
                break
            elif r > 0:
                reward = max(reward, r * backoff_ratio/(max_backoff_ratio+1e-8))
        c = 0 if len(constraint_violations) == 0 else np.mean(constraint_violations)
        self.trajectory.append([
            agent_features,
            agent_neighbor_features,
            list(neighbor_features.values()),
            actions, reward, c
        ])
        self.reward_log.append(reward - self.model.lambda_.item() * c)

        self.step_transition = {}

    def store_trajectory(self):
        self.transition_log.sort(key=lambda x: x["sim_time"])
        # build trajectory
        current_time = self.transition_log[0]["sim_time"]
        for trans in self.transition_log:
            timestamp = trans["sim_time"]
            if timestamp - current_time < 2e-6:
                # assert trans["node_id"] not in self.step_transition, "Duplicate node id."
                print('TODO: Duplicate node id.')
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

    def learn(self, rollback=False):
        if not self.enable_learning:
            return
        
        lambda_constraint = 0
        for _, c_log in self.constraint_log.items():
            c = np.mean(c_log)
            lambda_constraint += c#+CONSTRAINT_VIOLATION_TH if c < -CONSTRAINT_VIOLATION_TH or c > 0 else 0

        # rollback to latest model if high volatility
        if rollback:
            self.model.load_model(actor_dir=tmp_actor_dir, noise=0.01)
        else:
            tmp_actor_dir = os.path.join(self.save_dir, f'../../mappo_{self.agent_type.name}_actor.pt')
            self.model.save_model(actor_dir=tmp_actor_dir)
            self.model.learn(lambda_constraint)

        # update local models
        for _, agent in self.local_agents.items():
            agent.model.actor.load_state_dict(self.model.actor.state_dict())

        # update lambda
        if self.episode < 50:
            self.model.update_lambda(0)
        self.lambda_log.append(self.model.lambda_.item())

        # handle learning results
        actor_loss, critic_loss = np.mean(self.model.loss_log, axis=0) if len(self.model.loss_log) > 0 else (0, 0)
        self.loss_log.append((actor_loss, critic_loss))
        ave_entropy, ave_kl = np.mean(self.model.entropy_log, axis=0) if len(self.model.entropy_log) > 0 else (0, 0)
        self.entropy_log.append((ave_entropy, ave_kl))
        if self.debug_mode:
            self.plot_loss()
        self.model.reset()

    def reset(self):
        self.sim_time = 0
        self.step_time = []

        # organize trajectory
        self.store_trajectory()

        # record log for plot
        self.plot_results()
        if len(self.reward_log):
            self.episode += 1
            self.episode_log.append([np.sum(self.reward_log),
                                     np.sum(self.reward_log) / np.sum(np.array(self.reward_log) != 0)])
        
        ave_backoff_ratio = []
        for _, agent in self.local_agents.items():
            if len(agent.backoff_ratio_log) > 0:
                ave_backoff_ratio.append(np.mean(agent.backoff_ratio_log))
        self.ave_backoff_ratio_log.append(np.mean(ave_backoff_ratio))
        self.n_agents_log.append(len(ave_backoff_ratio))

        # train model
        self.learn(rollback=len(ave_backoff_ratio)==0)

        # reset trajectory log
        self.reward_log = []
        self.transition_log = []
        self.constraint_log = {}
        self.trajectory = []
        self.step_transition = {}

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
        fig, ax1 = plt.subplots()
        ax1.plot(range(len(actor_loss)), actor_loss, 'b-', linewidth=1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Actor Loss', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(range(len(critic_loss)), critic_loss, 'r-', linewidth=1)
        ax2.set_ylabel('Critic Loss', color='r')
        ax2.tick_params('y', colors='r')
        fig.savefig('{}/mappo_{}_loss.png'.format(self.save_dir, self.agent_type.name))
        plt.close(fig)

    def plot_episodes(self, save_dir=None):
        save_dir = save_dir if save_dir is not None else self.save_dir
        
        if len(self.episode_log) == 0:
            return
        
        tot_reward, ave_reward = zip(*self.episode_log)
        fig, ax1 = plt.subplots()
        ax1.plot(ave_reward, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward per TX', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        # ax2.plot(tot_reward, 'r-', linewidth=1)
        # ax2.set_ylabel('Total Reward', color='r')
        # ax2.tick_params('y', colors='r')
        ax2.plot(self.n_agents_log, 'r:', linewidth=1, alpha=0.5)
        ax2.set_ylabel('# of Agents', color='r')
        ax2.tick_params('y', colors='r')
        fig.savefig('{}/mappo_{}_reward.png'.format(save_dir, self.agent_type.name))
        plt.close()

        plt.figure()
        plt.plot(self.ave_backoff_ratio_log, 'g-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Average Backoff Ratio')
        plt.savefig('{}/mappo_{}_ave_backoff.png'.format(save_dir, self.agent_type.name))
        plt.close()

        plt.figure()
        plt.plot(self.lambda_log, 'r-', linewidth=1)
        plt.xlabel('Episode')
        plt.ylabel('Lambda')
        plt.savefig('{}/mappo_{}_lambda.png'.format(save_dir, self.agent_type.name))
        plt.close()

        actor_loss, critic_loss = zip(*self.loss_log)
        fig, ax1 = plt.subplots()
        ax1.plot(actor_loss, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Actor Loss', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(critic_loss, 'r-', linewidth=1)
        ax2.set_ylabel('Critic Loss', color='r')
        ax2.tick_params('y', colors='r')
        fig.savefig('{}/mappo_{}_loss.png'.format(save_dir, self.agent_type.name))
        plt.close()

        ave_entropy, ave_kl = zip(*self.entropy_log)
        fig, ax1 = plt.subplots()
        ax1.plot(ave_entropy, 'b-', linewidth=1)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Average Entropy', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(ave_kl, 'r-', linewidth=1)
        ax2.set_ylabel('Average KL Divergence', color='r')
        ax2.tick_params('y', colors='r')
        fig.tight_layout()
        fig.savefig('{}/mappo_{}_entropy.png'.format(save_dir, self.agent_type.name))
        plt.close()

        with open(os.path.join(save_dir, 'episode.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.n_agents_log)
            writer.writerow(tot_reward)
            writer.writerow(ave_reward)
            writer.writerow(self.ave_backoff_ratio_log)
            writer.writerow(self.lambda_log)