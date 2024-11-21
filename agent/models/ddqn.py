# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
GAMMA = 0.99
BATCH_SIZE = 32
MEMORY_SIZE = 2000
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQUENCY = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99
HIDDEN_SIZE = 64

# Neural Network for Q-value approximation
class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Replay Memory to store experiences
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN
class DQN:
    def __init__(self, state_size, action_size,
                 hidden_size=HIDDEN_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON_START
        self.eval_net = QNet(state_size, action_size, hidden_size)
        self.target_net = QNet(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.update_target_network()
        self.steps_done = 0
        self.loss_log = []

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def choose_action(self, state, action_mask=None):
        if action_mask is None:
            action_mask = [True]*self.action_size

        if random.random() > self.epsilon: # greedy
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                action_mask = torch.BoolTensor(action_mask).unsqueeze(0)
                q_values = self.eval_net(state)
                q_values[~action_mask] = -float('inf') # mask out illegal actions
                return q_values.max(1)[1].item()
        else:
            valid_actions = np.where(action_mask)[0]
            return random.choice(valid_actions)
    
    def store_transition(self, state, action, reward, next_state, action_mask=None):
        if action_mask is None:
            action_mask = [True]*self.action_size
        self.memory.push((state, action, reward, next_state, action_mask))

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = self.memory.sample(BATCH_SIZE)
        s, a, r, s_, mask = zip(*minibatch)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        mask = torch.BoolTensor(mask)

        q_values = self.eval_net(s).gather(1, a)
        next_q_values = self.target_net(s_).max(1, True)[0].detach()    # assume no mask for next action
        target_q_values = r + GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        if self.steps_done % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

        self.steps_done += 1
        self.loss_log.append(loss.item())
    
    def save_model(self, filename):
        torch.save(self.eval_net.state_dict(), filename)
    
    def load_model(self, filename):
        self.eval_net.load_state_dict(torch.load(filename))
        self.update_target_network()


# DDQN
class DDQN(DQN):
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = self.memory.sample(BATCH_SIZE)
        s, a, r, s_, mask = zip(*minibatch)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)
        mask = torch.BoolTensor(mask)

        q_values = self.eval_net(s).gather(1, a)
        a_ = self.eval_net(s_).max(1, True)[1]  # assume no mask for next action
        next_q_values = self.target_net(s_).gather(1, a_).detach()
        target_q_values = r + GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon *= EPSILON_DECAY

        if self.steps_done % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

        self.steps_done += 1
        self.loss_log.append(loss.item())