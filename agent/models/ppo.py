# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    

class ValueNet(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)
    

class PPO:
    def __init__(self, state_size, action_size, hidden_size, 
                 actor_lr, critic_lr, lmbda, n_epochs, eps, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(state_size, action_size, 
                                         hidden_size).to(self.device)
        self.critic = ValueNet(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.action_size = action_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.n_epochs = n_epochs
        self.eps = eps

        self.transition_dict = []
        self.loss_log = []
    
    def choose_action(self, state, action_mask=None):
        if action_mask is None:
            action_mask = [True]*self.action_size

        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state) * torch.tensor([action_mask], dtype=torch.float, device=self.device)
        probs = probs / probs.sum()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def store_transition(self, state, action, reward, next_state, action_mask=None):
        if action_mask is None:
            action_mask = [True]*self.action_size
        
        self.transition_dict.append((state, action, reward, next_state, action_mask))

    def compute_advantage(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in reversed(td_delta):
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float)

    def learn(self):
        s, a, r, s_, mask = zip(*self.transition_dict)
        states = torch.tensor(s, dtype=torch.float, device=self.device)
        actions = torch.tensor(a, device=self.device).unsqueeze(1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).unsqueeze(1)
        next_states = torch.tensor(s_, dtype=torch.float, device=self.device)

        td_target = rewards + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(td_delta.cpu()).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        for _ in range(self.n_epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            # entropy_bonus = -torch.mean(torch.distributions.Categorical(logits=self.actor(states)).entropy())
            # actor_loss += 0.01 * entropy_bonus
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.loss_log.append((actor_loss.item(), critic_loss.item()))
        
        self.transition_dict = []   # on policy
    
    def reset(self):
        self.loss_log = []
    
    def update_learning_rate(self, actor_lr, critic_lr):
        self.actor_optimizer.param_groups[0]['lr'] = actor_lr
        self.critic_optimizer.param_groups[0]['lr'] = critic_lr
    
    def save_model(self, actor_dir, critic_dir):
        torch.save(self.actor.state_dict(), actor_dir)
        torch.save(self.critic.state_dict(), critic_dir)
    
    def load_model(self, actor_dir, critic_dir):
        self.actor.load_state_dict(torch.load(actor_dir))
        self.critic.load_state_dict(torch.load(critic_dir))