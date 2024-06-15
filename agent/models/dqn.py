import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
HIDDEN_SIZE = 64
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 2000
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQUENCY = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1000

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

    def push(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

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

    def choose_action(self, state):
        if random.random() > self.epsilon: # greedy
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.eval_net(state)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = self.memory.sample(BATCH_SIZE)
        s, a, r, s_ = zip(*minibatch)

        s = torch.FloatTensor(s)
        a = torch.LongTensor(a).unsqueeze(1)
        r = torch.FloatTensor(r).unsqueeze(1)
        s_ = torch.FloatTensor(s_)

        q_values = self.eval_net(s).gather(1, a)
        next_q_values = self.target_net(s_).max(1, True)[0].detach()
        target_q_values = r + GAMMA * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > EPSILON_END:
            self.epsilon -= (EPSILON_START - EPSILON_END) / EPSILON_DECAY

        if self.steps_done % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()

        self.steps_done += 1
        self.loss_log.append(loss.item())