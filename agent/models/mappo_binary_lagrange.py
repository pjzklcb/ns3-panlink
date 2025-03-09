# Copyright (c) 2025
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
    def __init__(self, agent_feature_dim, neighbor_feature_dim, hidden_dim, pooling="mean"):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(agent_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, agent_features, neighbor_features, key_padding_mask=None):
        # agent_features: (batch_size, agent_feature_dim)
        # neighbor_features: (batch_size, n_neighbors, neighbor_feature_dim)

        batch_size, n_neighbors, _ = neighbor_features.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, n_neighbors, 
                                           dtype=torch.bool,
                                           device=neighbor_features.device)
            
        # get embeddings
        agent_embeds = F.tanh(self.fc1(agent_features))  # (batch_size, hidden_dim)
        neighbor_embeds = F.tanh(self.fc2(neighbor_features))  # (batch_size, n_neighbors, hidden_dim)

        # Pooling neighbor_embeds
        if self.pooling == "mean":
            valid_count = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_count
        elif self.pooling == "max":
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([agent_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, 2 * hidden_dim)
        return F.sigmoid(self.fc3(x))
    

class ValueNet(nn.Module):
    def __init__(self, agent_feature_dim, neighbor_feature_dim, hidden_dim, pooling="mean"):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(agent_feature_dim, hidden_dim)
        self.fc2 = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, agent_features, neighbor_features, agent_key_padding_mask=None, neighbor_key_padding_mask=None):
        # agent_features: (batch_size, n_agents, agent_feature_dim)
        # neighbor_features: (batch_size, n_neighbors, neighbor_feature_dim)

        batch_size, n_agents, _ = agent_features.shape
        _, n_neighbors, _ = neighbor_features.shape
        if agent_key_padding_mask is None:
            agent_key_padding_mask = torch.zeros(batch_size, n_agents, 
                                                 dtype=torch.bool,
                                                 device=agent_features.device)
        if neighbor_key_padding_mask is None:
            neighbor_key_padding_mask = torch.zeros(batch_size, n_neighbors, 
                                                   dtype=torch.bool,
                                                   device=neighbor_features.device)
            
        # get embeddings
        agent_embeds = F.tanh(self.fc1(agent_features))  # (batch_size, n_agents, hidden_dim)
        neighbor_embeds = F.tanh(self.fc2(neighbor_features))  # (batch_size, seq_len, hidden_dim)

        # Pooling embeddings
        if self.pooling == "mean":
            valid_agents = (~agent_key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_agent_embeds = agent_embeds.masked_fill(agent_key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_agents
            valid_neighbors = (~neighbor_key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(neighbor_key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_neighbors
        elif self.pooling == "max":
            pooled_agent_embeds = agent_embeds.masked_fill(agent_key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(neighbor_key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([pooled_agent_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, 2 * hidden_dim)
        return self.fc3(x)
    

class MAPPO:
    def __init__(self, agent_feature_dim, neighbor_feature_dim, hidden_dim, 
                 actor_lr, critic_lr, lmbda, n_epochs, eps, gamma, lambda_, eta_lambda):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(agent_feature_dim, neighbor_feature_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(agent_feature_dim, neighbor_feature_dim, hidden_dim).to(self.device)
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
    
    def choose_action(self, agent_features, neighbor_features):
        agent_features = torch.tensor([agent_features], dtype=torch.float).to(self.device)
        neighbor_features = torch.tensor([neighbor_features], dtype=torch.float).to(self.device)
        probs = self.actor(agent_features, neighbor_features)
        return probs.item()

    def store_transition(self, agent_features, agent_neighbor_features, neighbor_features, 
                         action, reward, constraint_violation, next_agent_features, next_neighbor_features):
        self.transition_dict.append((agent_features, agent_neighbor_features, neighbor_features, 
                                     action, reward, constraint_violation, next_agent_features, next_neighbor_features))

    def compute_advantage(self, td_delta):
        advantage = torch.zeros_like(td_delta)
        advantage[-1] = td_delta[-1]
        for t in range(len(td_delta)-2, -1, -1):
            advantage[t] = td_delta[t] + self.gamma * self.lmbda * advantage[t+1]
        return advantage

    def learn(self):
        u, vu, v, a, r, c, u_, v_ = zip(*self.transition_dict)
        agent_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in u], batch_first=True
        )   # (batch_size, max_n_agents, agent_feature_dim)
        agent_features_cat = torch.tensor(
            [y for x in u for y in x], dtype=torch.float, device=self.device
        )   # (batch_size*n_agents, agent_feature_dim)
        agent_neighbor_features_cat = rnn_utils.pad_sequence(
            [torch.tensor(y, dtype=torch.float, device=self.device) for x in vu for y in x], batch_first=True
        )   # (batch_size*n_agents, max_n_neighbors, neighbor_feature_dim)
        neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in v], batch_first=True
        )   # (batch_size, max_n_neighbors, neighbor_feature_dim)
        actions_cat = torch.tensor(
            [y for x in a for y in x], dtype=torch.float, device=self.device
        ).unsqueeze(1)   # (batch_size*n_agents, 1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).unsqueeze(1)               # (batch_size, 1)
        constraint_violations = torch.tensor(c, dtype=torch.float, device=self.device).unsqueeze(1) # (batch_size, 1)
        next_agent_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in u_], batch_first=True
        )
        next_neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in v_], batch_first=True
        )

        # generate padding mask
        agent_mask = torch.all(torch.eq(agent_features, 0), dim=-1)         # (batch_size, max_n_agents)
        next_agent_mask = torch.all(torch.eq(next_agent_features, 0), dim=-1)
        neighbor_mask = torch.all(torch.eq(neighbor_features, 0), dim=-1)   # (batch_size, max_n_neighbors)
        next_neighbor_mask = torch.all(torch.eq(next_neighbor_features, 0), dim=-1)
        agent_neighbor_mask = torch.all(torch.eq(agent_neighbor_features_cat, 0), dim=-1) # (batch_size*n_agents, max_n_neighbors)
        n_agents = (~agent_mask).sum(dim=1)                                    # (batch_size,)

        # compensate for constraint violations
        rewards -= self.lambda_ * constraint_violations

        # update actor-critic
        td_target = rewards + self.gamma * self.critic(next_agent_features, next_neighbor_features, 
                                                       next_agent_mask, next_neighbor_mask)
        td_delta = td_target - self.critic(agent_features, neighbor_features, agent_mask, neighbor_mask)
        advantage = self.compute_advantage(td_delta.detach()).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        advantage_cat = torch.repeat_interleave(advantage, n_agents, dim=0) # (batch_size*n_agents,)

        old_probs = self.actor(agent_features_cat, agent_neighbor_features_cat, agent_neighbor_mask).detach()
        old_log_probs = torch.where(actions_cat == 1, torch.log(old_probs), torch.log(1 - old_probs))
        for _ in range(self.n_epochs):
            probs = self.actor(agent_features_cat, agent_neighbor_features_cat, agent_neighbor_mask)
            log_probs = torch.where(actions_cat == 1, torch.log(probs), torch.log(1 - probs))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage_cat
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage_cat
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            critic_loss = torch.mean(F.mse_loss(self.critic(agent_features, neighbor_features, agent_mask, neighbor_mask), 
                                                td_target.detach()))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
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