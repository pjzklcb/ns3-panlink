# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class PolicyNet(nn.Module):
    def __init__(self, self_dim, neighbor_dim, hidden_dim, pooling="mean"):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(self_dim, hidden_dim)
        self.fc2 = nn.Linear(neighbor_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, self_features, neighbor_features, key_padding_mask=None):
        # self_features: (batch_size, self_dim)
        # neighbor_features: (batch_size, seq_len, neighbor_dim)

        batch_size, num_neighbors, _ = neighbor_features.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, num_neighbors, 
                                           dtype=torch.bool,
                                           device=neighbor_features.device)
            
        # get embeddings
        self_embeds = F.tanh(self.fc1(self_features))  # (batch_size, self_dim)
        neighbor_embeds = F.tanh(self.fc2(neighbor_features))  # (batch_size, seq_len, hidden_dim)

        # Pooling neighbor_embeds
        if self.pooling == "mean":
            valid_count = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_count
        elif self.pooling == "max":
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([self_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, 2 * hidden_dim)
        return F.sigmoid(self.fc3(x))
    

class ValueNet(nn.Module):
    def __init__(self, self_dim, neighbor_dim, hidden_dim, pooling="mean"):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(self_dim, hidden_dim)
        self.fc2 = nn.Linear(neighbor_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, self_features, neighbor_features, key_padding_mask=None):
        # self_features: (batch_size, self_dim)
        # neighbor_features: (batch_size, seq_len, neighbor_dim)

        batch_size, num_neighbors, _ = neighbor_features.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, num_neighbors, 
                                           dtype=torch.bool,
                                           device=neighbor_features.device)
            
        # get embeddings
        self_embeds = F.tanh(self.fc1(self_features))  # (batch_size, self_dim)
        neighbor_embeds = F.tanh(self.fc2(neighbor_features))  # (batch_size, seq_len, hidden_dim)

        # Pooling neighbor_embeds
        if self.pooling == "mean":
            valid_count = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_count
        elif self.pooling == "max":
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([self_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, 2 * hidden_dim)
        return self.fc3(x)
    

class PPO:
    def __init__(self, self_dim, neighbor_dim, hidden_dim, 
                 actor_lr, critic_lr, lmbda, n_epochs, eps, gamma, lambda_, eta_lambda):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(self_dim, neighbor_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(self_dim, neighbor_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.n_epochs = n_epochs
        self.eps = eps
        self.lambda_ = lambda_
        self.eta_lambda = eta_lambda

        self.transition_dict = []
        self.loss_log = []
    
    def choose_action(self, self_features, neighbor_features):
        self_features = torch.tensor([self_features], dtype=torch.float).to(self.device)
        neighbor_features = torch.tensor([neighbor_features], dtype=torch.float).to(self.device)
        probs = self.actor(self_features, neighbor_features)
        return probs.item()

    def store_transition(self, state, action, reward, constraint_violation, next_state):
        self_features, neighbor_features = state
        next_self_features, next_neighbor_features = next_state
        self.transition_dict.append((self_features, neighbor_features, action, reward, 
                                     constraint_violation, next_self_features, next_neighbor_features))

    def compute_advantage(self, td_delta):
        advantage = torch.zeros_like(td_delta)
        advantage[-1] = td_delta[-1]
        for t in range(len(td_delta)-2, -1, -1):
            advantage[t] = td_delta[t] + self.gamma * self.lmbda * advantage[t+1]
        return advantage

    def learn(self):
        s, n, a, r, c, s_, n_ = zip(*self.transition_dict)
        self_features = torch.tensor(s, dtype=torch.float, device=self.device)
        neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in n], batch_first=True
        )
        actions = torch.tensor(a, device=self.device).unsqueeze(1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).unsqueeze(1)
        constraint_violations = torch.tensor(c, dtype=torch.float, device=self.device).unsqueeze(1)
        next_self_features = torch.tensor(s_, dtype=torch.float, device=self.device)
        next_neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in n_], batch_first=True
        )

        # generate padding mask
        neighbor_mask = torch.all(torch.eq(neighbor_features, 0), dim=-1)
        next_neighbor_mask = torch.all(torch.eq(next_neighbor_features, 0), dim=-1)

        # compensate for constraint violations
        rewards -= self.lambda_ * constraint_violations

        # update actor-critic
        td_target = rewards + self.gamma * self.critic(next_self_features, next_neighbor_features, next_neighbor_mask)
        td_delta = td_target - self.critic(self_features, neighbor_features, neighbor_mask)
        advantage = self.compute_advantage(td_delta.detach()).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        old_probs = self.actor(self_features, neighbor_features, neighbor_mask).detach()
        old_log_probs = torch.where(actions == 1, torch.log(old_probs), torch.log(1 - old_probs))

        for _ in range(self.n_epochs):
            probs = self.actor(self_features, neighbor_features, neighbor_mask)
            log_probs = torch.where(actions == 1, torch.log(probs), torch.log(1 - probs))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(self_features, neighbor_features, neighbor_mask), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.loss_log.append((actor_loss.item(), critic_loss.item()))
        
        # update lagrangian multiplier
        avg_constraint = torch.mean(constraint_violations)
        self.lambda_ += self.eta_lambda * avg_constraint.item()
        self.lambda_ = min(1, max(0, self.lambda_))

        self.transition_dict = []   # on policy
    
    def reset(self):
        self.loss_log = []
    
    def update_learning_rate(self, actor_lr, critic_lr, eta_lambda):
        self.actor_optimizer.param_groups[0]['lr'] = actor_lr
        self.critic_optimizer.param_groups[0]['lr'] = critic_lr
        self.eta_lambda = eta_lambda
    
    def save_model(self, actor_dir, critic_dir):
        torch.save(self.actor.state_dict(), actor_dir)
        torch.save(self.critic.state_dict(), critic_dir)
    
    def load_model(self, actor_dir, critic_dir):
        self.actor.load_state_dict(torch.load(actor_dir))
        self.critic.load_state_dict(torch.load(critic_dir))
        self.actor.to(self.device)
        self.critic.to(self.device)