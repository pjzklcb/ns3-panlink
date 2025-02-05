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
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc5 = nn.Linear(hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, self_features, neighbor_features, seq_len_mask=None, n_neighbor_mask=None):
        """
        Args:
            self_features: (batch_size, seq_len, self_dim)
            neighbor_features: (batch_size, n_neighbors, seq_len, neighbor_dim)
            seq_len_mask: (batch_size,)
            n_neighbor_mask: (batch_size,)
        """
        batch_size, n_neighbors, seq_len, _ = neighbor_features.shape
        key_padding_mask = (
            torch.arange(n_neighbors, 
                         device=neighbor_features.device).expand(batch_size, -1) >= n_neighbor_mask.unsqueeze(1)
        )
            
        # get embeddings
        self_embeds = F.tanh(self.fc1(self_features))
        neighbor_embeds = F.tanh(self.fc2(neighbor_features)).view(batch_size*n_neighbors, seq_len, -1)

        if seq_len_mask is None:
            self_embeds, _ = self.gru1(self_embeds)
            self_embeds = F.tanh(self.fc3(self_embeds[:, -1, :]))  # (batch_size, hidden_dim / 2)
            neighbor_embeds, _ = self.gru2(neighbor_embeds)
            neighbor_embeds = F.tanh(self.fc4(neighbor_embeds[:, -1, :])).view(batch_size, n_neighbors, -1)
        else:
            self_embeds = rnn_utils.pack_padded_sequence(self_embeds, seq_len_mask, batch_first=True)
            self_embeds, _ = self.gru1(self_embeds)
            self_embeds, _ = rnn_utils.pad_packed_sequence(self_embeds, batch_first=True)
            self_embeds = F.tanh(self.fc3(self_embeds[torch.arange(batch_size), seq_len_mask - 1, :]))  # (batch_size, hidden_dim / 2)
            
            extend_seq_len_mask = seq_len_mask.unsqueeze(-1).expand(-1, n_neighbors).reshape(-1)
            neighbor_embeds = rnn_utils.pack_padded_sequence(neighbor_embeds, extend_seq_len_mask, batch_first=True)
            neighbor_embeds, _ = self.gru2(neighbor_embeds)
            neighbor_embeds, _ = rnn_utils.pad_packed_sequence(neighbor_embeds, batch_first=True)
            neighbor_embeds = F.tanh(self.fc4(neighbor_embeds[torch.arange(batch_size), seq_len_mask - 1, :])).view(batch_size, n_neighbors, -1)

        # Pooling neighbor_embeds
        if self.pooling == "mean":
            valid_count = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_count
        elif self.pooling == "max":
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([self_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, hidden_dim)
        return F.sigmoid(self.fc5(x))
    

class ValueNet(nn.Module):
    def __init__(self, self_dim, neighbor_dim, hidden_dim, pooling="mean"):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(self_dim, hidden_dim)
        self.fc2 = nn.Linear(neighbor_dim, hidden_dim)
        self.gru1 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc5 = nn.Linear(hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, self_features, neighbor_features, seq_len_mask=None, n_neighbor_mask=None):
        """
        Args:
            self_features: (batch_size, seq_len, self_dim)
            neighbor_features: (batch_size, n_neighbors, seq_len, neighbor_dim)
            seq_len_mask: (batch_size,)
            n_neighbor_mask: (batch_size,)
        """
        batch_size, n_neighbors, seq_len, _ = neighbor_features.shape
        key_padding_mask = torch.zeros(batch_size, n_neighbors, dtype=torch.bool, 
                                       device=neighbor_features.device)
        if n_neighbor_mask is not None:
            for i, count in enumerate(n_neighbor_mask):
                key_padding_mask[i, count:] = True
            
        # get embeddings
        self_embeds = F.tanh(self.fc1(self_features))
        neighbor_embeds = F.tanh(self.fc2(neighbor_features)).view(batch_size*n_neighbors, seq_len, -1)

        if seq_len_mask is None:
            self_embeds, _ = self.gru1(self_embeds)
            self_embeds = F.tanh(self.fc3(self_embeds[:, -1, :]))  # (batch_size, hidden_dim / 2)
            neighbor_embeds, _ = self.gru2(neighbor_embeds)
            neighbor_embeds = F.tanh(self.fc4(neighbor_embeds[:, -1, :])).view(batch_size, n_neighbors, -1)
        else:
            self_embeds = rnn_utils.pack_padded_sequence(self_embeds, seq_len_mask, batch_first=True, enforce_sorted=False)
            self_embeds, _ = self.gru1(self_embeds)
            self_embeds, _ = rnn_utils.pad_packed_sequence(self_embeds, batch_first=True)
            self_embeds = F.tanh(self.fc3(self_embeds[torch.arange(batch_size), seq_len_mask - 1, :]))  # (batch_size, hidden_dim / 2)
            
            extend_seq_len_mask = seq_len_mask.unsqueeze(-1).expand(-1, n_neighbors).reshape(-1)
            neighbor_embeds = rnn_utils.pack_padded_sequence(neighbor_embeds, extend_seq_len_mask, batch_first=True, enforce_sorted=False)
            neighbor_embeds, _ = self.gru2(neighbor_embeds)
            neighbor_embeds, _ = rnn_utils.pad_packed_sequence(neighbor_embeds, batch_first=True)
            neighbor_embeds = F.tanh(self.fc4(neighbor_embeds[torch.arange(batch_size), seq_len_mask - 1, :])).view(batch_size, n_neighbors, -1)

        # Pooling neighbor_embeds
        if self.pooling == "mean":
            valid_count = (~key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_count
        elif self.pooling == "max":
            pooled_neighbor_embeds = neighbor_embeds.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        x = torch.cat([self_embeds, pooled_neighbor_embeds], dim=-1)  # (batch_size, hidden_dim)
        return self.fc5(x)
    

class PPO:
    def __init__(self, self_dim, neighbor_dim, hidden_dim, 
                 actor_lr, critic_lr, lmbda, n_epochs, eps, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(self_dim, neighbor_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(self_dim, neighbor_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.n_epochs = n_epochs
        self.eps = eps

        self.transition_dict = []
        self.loss_log = []
    
    def choose_action(self, next_state):
        self_features, neighbor_features = next_state
        self_features = torch.tensor([self_features], dtype=torch.float).to(self.device)
        neighbor_features = torch.tensor([neighbor_features], dtype=torch.float).to(self.device)
        probs = self.actor(self_features, neighbor_features)
        return probs.item()

    def store_transition(self, state, action, reward, next_state):
        self_features, neighbor_features = state
        next_self_features, next_neighbor_features = next_state
        self.transition_dict.append((self_features, neighbor_features, action, reward, next_self_features, next_neighbor_features))

    def compute_advantage(self, td_delta):
        advantage = torch.zeros_like(td_delta)
        advantage[-1] = td_delta[-1]
        for t in range(len(td_delta)-2, -1, -1):
            advantage[t] = td_delta[t] + self.gamma * self.lmbda * advantage[t+1]
        return advantage

    def learn(self):
        s, n, a, r, s_, n_ = zip(*self.transition_dict)
        self_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in s], batch_first=True
        ) # (batch_size, seq_len, self_dim)
        neighbor_features = rnn_utils.pad_sequence(
            [F.pad(torch.tensor(x, dtype=torch.float, device=self.device), 
                   (0, 0, self_features.shape[1] - len(x[0]), 0), mode='constant', value=0) for x in n], 
            batch_first=True
        ) # (batch_size, n_neighbors, seq_len, neighbor_dim)
        actions = torch.tensor(a, device=self.device).unsqueeze(1)
        rewards = torch.tensor(r, dtype=torch.float, device=self.device).unsqueeze(1)
        next_self_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float, device=self.device) for x in s_], batch_first=True
        )
        next_neighbor_features = rnn_utils.pad_sequence(
            [F.pad(torch.tensor(x, dtype=torch.float, device=self.device), 
                   (0, 0, self_features.shape[1] - len(x[0]), 0), mode='constant', value=0) for x in n_], 
            batch_first=True
        )

        # generate padding mask
        seq_len_mask = torch.tensor([len(x) for x in s], device='cpu')
        n_neighbor_mask = torch.tensor([len(x) for x in n], device='cpu')
        next_seq_len_mask = torch.tensor([len(x) for x in s_], device='cpu')
        next_n_neighbor_mask = torch.tensor([len(x) for x in n_], device='cpu')

        td_target = rewards + self.gamma * self.critic(next_self_features, next_neighbor_features, 
                                                       next_seq_len_mask, next_n_neighbor_mask)
        # rewards_centered = rewards - rewards.mean()
        # td_target = rewards_centered + self.gamma * self.critic(next_states)
        td_delta = td_target - self.critic(self_features, neighbor_features, seq_len_mask, n_neighbor_mask)
        advantage = self.compute_advantage(td_delta.detach()).to(self.device)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        old_probs = self.actor(self_features, neighbor_features, seq_len_mask, n_neighbor_mask).detach()
        old_log_probs = torch.where(actions == 1, torch.log(old_probs), torch.log(1 - old_probs))

        for _ in range(self.n_epochs):
            probs = self.actor(self_features, neighbor_features, seq_len_mask, n_neighbor_mask)
            log_probs = torch.where(actions == 1, torch.log(probs), torch.log(1 - probs))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(self_features, neighbor_features, seq_len_mask, n_neighbor_mask), 
                                                td_target.detach()))
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
        self.actor.to(self.device)
        self.critic.to(self.device)