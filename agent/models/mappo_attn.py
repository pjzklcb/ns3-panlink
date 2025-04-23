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
import numpy as np

N_EPOCHS = 10
BATCH_SIZE = 4096
ENTROPY_COEF = 0.01
EPS = 0.2
DROPOUT = 0.2


# reshape concatenated features and padding
def restore_cat(feature_cat, n_agents_per_sample):
    # feature_cat: (batch_size*n_agents, feature_dim)

    feature_split = torch.split(
        feature_cat, n_agents_per_sample, dim=0
    ) # a list of (n_agents_per_sample, feature_dim)
    feature_pad = rnn_utils.pad_sequence(
        feature_split, batch_first=True
    ) # (batch_size, max_n_agents, feature_dim)

    return feature_pad


class PolicyNet(nn.Module):
    def __init__(self, agent_feature_dim, neighbor_feature_dim, action_dim, hidden_dim, n_head=1):
        super(PolicyNet, self).__init__()
        self.query = nn.Linear(agent_feature_dim, hidden_dim)
        self.key = nn.Linear(agent_feature_dim, hidden_dim)
        self.value = nn.Linear(agent_feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(DROPOUT)

        self.neighbor_key = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.neighbor_value = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Linear(hidden_dim, action_dim)

    def forward(self, agent_features, neighbor_features, key_padding_mask=None):
        # agent_features: (batch_size, agent_feature_dim)
        # neighbor_features: (batch_size, n_neighbor, neighbor_feature_dim)

        batch_size, n_neighbor, _ = neighbor_features.shape
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, n_neighbor, 
                                           dtype=torch.bool,
                                           device=neighbor_features.device)
            
        Q = self.dropout1(self.query(agent_features).unsqueeze(1))  # (batch_size, 1, hidden_dim)
        K = torch.cat([self.dropout1(self.key(agent_features).unsqueeze(1)), 
                       self.dropout1(self.neighbor_key(neighbor_features))], dim=1)
        V = torch.cat([self.dropout1(self.value(agent_features).unsqueeze(1)), 
                       self.dropout1(self.neighbor_value(neighbor_features))], dim=1)
        M = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool).to(key_padding_mask.device), 
                       key_padding_mask], dim=1)
        
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=M)  # (batch_size, 1, hidden_dim)
        attn_output = self.layer_norm(self.dropout2(attn_output + Q))

        return F.softmax(self.decoder(attn_output.squeeze(1)), dim=-1)  # (batch_size, action_dim)
    

class ValueNet(nn.Module):
    def __init__(self, agent_feature_dim, neighbor_feature_dim, hidden_dim, n_head=1, pooling="mean"):
        super(ValueNet, self).__init__()
        self.query = nn.Linear(agent_feature_dim, hidden_dim)
        self.key = nn.Linear(agent_feature_dim, hidden_dim)
        self.value = nn.Linear(agent_feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(DROPOUT)

        self.neighbor_key = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.neighbor_value = nn.Linear(neighbor_feature_dim, hidden_dim)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Linear(hidden_dim, 1)

        # Pooling method: mean or max
        if pooling not in ["mean", "max"]:
            raise ValueError(f"Unsupported pooling method: {pooling}. Choose 'mean' or 'max'.")
        self.pooling = pooling

    def forward(self, agent_features, neighbor_features, agent_key_padding_mask=None, neighbor_key_padding_mask=None):
        # agent_features: (batch_size, n_agent, agent_feature_dim)
        # neighbor_features: (batch_size, n_neighbor, neighbor_feature_dim)

        batch_size, n_agent, _ = agent_features.shape
        _, n_neighbor, _ = neighbor_features.shape
        if agent_key_padding_mask is None:
            agent_key_padding_mask = torch.zeros(batch_size, n_agent, 
                                                 dtype=torch.bool,
                                                 device=agent_features.device)
        if neighbor_key_padding_mask is None:
            neighbor_key_padding_mask = torch.zeros(batch_size, n_neighbor, 
                                                   dtype=torch.bool,
                                                   device=neighbor_features.device)
            
        Q = self.dropout1(self.query(agent_features))  # (batch_size, n_agent, hidden_dim)
        K = torch.cat([self.dropout1(self.key(agent_features)), 
                       self.dropout1(self.neighbor_key(neighbor_features))], dim=1)
        V = torch.cat([self.dropout1(self.value(agent_features)), 
                       self.dropout1(self.neighbor_value(neighbor_features))], dim=1)
        M = torch.cat([agent_key_padding_mask, neighbor_key_padding_mask], dim=1)

        attn_output, _ = self.mha(Q, K, V, key_padding_mask=M)  # (batch_size, n_agent, hidden_dim)
        attn_output = self.layer_norm(self.dropout2(attn_output + Q)) 

        # Pooling embeddings
        if self.pooling == "mean":
            valid_agents = (~agent_key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
            pooled_attn_output = attn_output.masked_fill(agent_key_padding_mask.unsqueeze(-1), 0).sum(dim=1) / valid_agents
        elif self.pooling == "max":
            pooled_attn_output = attn_output.masked_fill(agent_key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]

        return self.decoder(pooled_attn_output)
    

class MAPPO:
    def __init__(self, agent_feature_dim, neighbor_feature_dim, action_dim, hidden_dim, 
                 actor_lr, critic_lr, lmbda, gamma, lambda_, lambda_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = PolicyNet(agent_feature_dim, neighbor_feature_dim, action_dim, hidden_dim).to(self.device)
        self.critic = ValueNet(agent_feature_dim, neighbor_feature_dim, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)

        self.gamma = gamma
        self.lmbda = lmbda
        self.lambda_ = nn.Parameter(torch.tensor(lambda_), requires_grad=True)
        self.lambda_optimizer = optim.Adam([self.lambda_], lr=lambda_lr, eps=1e-5)
        self.entropy_coef = ENTROPY_COEF

        self.transition_dict = []
        self.loss_log = []
        self.entropy_log = []
    
    def choose_action(self, agent_features, neighbor_features):
        agent_features = torch.tensor([agent_features], dtype=torch.float).to(self.device)
        neighbor_features = torch.tensor([neighbor_features], dtype=torch.float).to(self.device)
        probs = self.actor(agent_features, neighbor_features)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), probs[0][0].item()

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

    def learn(self, lambda_constraint):
        u, vu, v, a, r, c, u_, v_ = zip(*self.transition_dict)
        # transfer to cpu tensor
        agent_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float) for x in u], batch_first=True
        )   # (n_steps, max_n_agents, agent_feature_dim)
        agent_features_cat = torch.tensor(
            [y for x in u for y in x], dtype=torch.float
        )   # (n_steps*n_agents, agent_feature_dim)
        agent_neighbor_features_cat = rnn_utils.pad_sequence(
            [torch.tensor(y, dtype=torch.float) for x in vu for y in x], batch_first=True
        )   # (n_steps*n_agents, max_n_neighbors, neighbor_feature_dim)
        neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float) for x in v], batch_first=True
        )   # (n_steps, max_n_neighbors, neighbor_feature_dim)
        actions = rnn_utils.pad_sequence(
            [torch.tensor(x) for x in a], batch_first=True
        )   # (n_steps, max_n_agents, action_dim)
        rewards = torch.tensor(r, dtype=torch.float).unsqueeze(1)               # (n_steps, 1)
        constraint_violations = torch.tensor(c, dtype=torch.float).unsqueeze(1) # (n_steps, 1)
        next_agent_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float) for x in u_], batch_first=True
        )
        next_neighbor_features = rnn_utils.pad_sequence(
            [torch.tensor(x, dtype=torch.float) for x in v_], batch_first=True
        )

        # generate padding mask
        agent_mask = torch.all(torch.eq(agent_features, 0), dim=-1) # (n_steps, max_n_agents)
        agent_neighbor_cat_mask = torch.all(
            torch.eq(agent_neighbor_features_cat, 0), dim=-1
        ) # (n_steps*n_agents, max_n_neighbors)
        neighbor_mask = torch.all(torch.eq(neighbor_features, 0), dim=-1) # (n_steps, max_n_neighbors)
        next_agent_mask = torch.all(torch.eq(next_agent_features, 0), dim=-1)
        next_neighbor_mask = torch.all(torch.eq(next_neighbor_features, 0), dim=-1)
        n_agents_per_sample = (~agent_mask).sum(dim=1) # (n_steps,)

        # compensate for constraint violations
        rewards -= self.lambda_ * constraint_violations

        # calculate advantage
        try:
            self.actor = self.actor.cpu()
            self.critic = self.critic.cpu()
            td_target = rewards + self.gamma * self.critic(next_agent_features, next_neighbor_features, 
                                                           next_agent_mask, next_neighbor_mask)
            td_delta = td_target - self.critic(agent_features, neighbor_features, agent_mask, neighbor_mask)
            advantage = self.compute_advantage(td_delta.detach())
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            advantage_cat = torch.repeat_interleave(advantage, n_agents_per_sample, dim=0) # (n_steps*n_agents,)
            old_log_probs_cat = torch.log(self.actor(agent_features_cat, 
                                                     agent_neighbor_features_cat, 
                                                     agent_neighbor_cat_mask)).detach()
            
        finally:
            self.actor = self.actor.to(self.device)
            self.critic = self.critic.to(self.device)
        
        # reshape and padding to (n_steps, ...)
        n_agents_per_sample = n_agents_per_sample.cpu().tolist()
        agent_neighbor_features = restore_cat(agent_neighbor_features_cat, n_agents_per_sample)
        agent_neighbor_mask = restore_cat(agent_neighbor_cat_mask, n_agents_per_sample)
        advantage = restore_cat(advantage_cat, n_agents_per_sample)
        old_log_probs = restore_cat(old_log_probs_cat, n_agents_per_sample)

        # update actor-critic
        for _ in range(N_EPOCHS):
            indices = torch.randperm(len(u))
            for i in range(0, len(u), BATCH_SIZE):
                # create mini-batch
                batch_idx = indices[i:i+BATCH_SIZE]
                batch_u = agent_features[batch_idx].to(self.device)
                batch_vu = agent_neighbor_features[batch_idx].to(self.device) 
                batch_v = neighbor_features[batch_idx].to(self.device)
                batch_a = actions[batch_idx].to(self.device) 
                batch_u_mask = agent_mask[batch_idx].to(self.device)
                batch_vu_mask = agent_neighbor_mask[batch_idx].to(self.device)
                batch_v_mask = neighbor_mask[batch_idx].to(self.device)
                batch_td_target = td_target[batch_idx].detach().to(self.device)
                batch_advantage = advantage[batch_idx].to(self.device)
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device)

                # remove padding
                valid_batch_idx, valid_gent_idx = torch.where(~batch_u_mask)
                batch_u_cat = batch_u[valid_batch_idx, valid_gent_idx]
                batch_vu_cat = batch_vu[valid_batch_idx, valid_gent_idx]
                batch_a_cat = batch_a[valid_batch_idx, valid_gent_idx].unsqueeze(1)
                batch_vu_mask_cat = batch_vu_mask[valid_batch_idx, valid_gent_idx]
                batch_advantage_cat = batch_advantage[valid_batch_idx, valid_gent_idx]

                batch_old_log_probs_cat = batch_old_log_probs[valid_batch_idx, valid_gent_idx]
                batch_log_probs_cat = torch.log(self.actor(batch_u_cat, 
                                                           batch_vu_cat, 
                                                           batch_vu_mask_cat))
                # calculate entropy and kl divergence
                entropy = -torch.sum(batch_log_probs_cat * torch.exp(batch_log_probs_cat), dim=1).mean()
                kl = torch.sum(torch.exp(batch_old_log_probs_cat) * (batch_old_log_probs_cat - batch_log_probs_cat), dim=1).mean()

                batch_old_log_probs_cat_select = batch_old_log_probs_cat.gather(1, batch_a_cat)
                batch_log_probs_cat_select = batch_log_probs_cat.gather(1, batch_a_cat)
                ratio = torch.exp(batch_log_probs_cat_select - batch_old_log_probs_cat_select)
                surr1 = ratio * batch_advantage_cat
                surr2 = torch.clamp(ratio, 1 - EPS, 1 + EPS) * batch_advantage_cat
                actor_loss = -torch.mean(torch.min(surr1, surr2)) - self.entropy_coef * entropy
                critic_loss = torch.mean(F.mse_loss(self.critic(batch_u, batch_v, batch_u_mask, batch_v_mask), 
                                                    batch_td_target))
            
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                self.loss_log.append((actor_loss.item(), critic_loss.item()))
                self.entropy_log.append((entropy.item(), kl.item()))
        
        # adjust hyperparameters
        ave_entropy, ave_kl = np.mean(self.entropy_log, axis=0)
        actor_lr = self.actor_optimizer.param_groups[0]['lr']
        if ave_kl > 0.1:
            actor_lr *= 0.9
        elif ave_kl < 0.01:
            actor_lr *= 1.1
        self.actor_optimizer.param_groups[0]['lr'] = np.clip(actor_lr, 1e-5, 3e-4)

        # adjust entropy coefficient
        if ave_entropy < 0.5:
            self.entropy_coef *= 1.1
        elif ave_entropy > 1.5:
            self.entropy_coef *= 0.9
        self.entropy_coef = np.clip(self.entropy_coef, 0.01, 0.1)
        
        # update lagrangian multiplier
        lambda_loss = -self.lambda_ * lambda_constraint 
        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        with torch.no_grad():
            self.lambda_.data.clamp_(0, 1)

        self.transition_dict = []   # on policy
    
    def reset(self):
        self.loss_log = []
        self.entropy_log = []
    
    def update_learning_rate(self, actor_lr, critic_lr, lambda_lr):
        self.actor_optimizer.param_groups[0]['lr'] = actor_lr
        self.critic_optimizer.param_groups[0]['lr'] = critic_lr
        self.lambda_optimizer.param_groups[0]['lr'] = lambda_lr
    
    def update_lambda(self, new_lambda):
        with torch.no_grad():
            self.lambda_.data = torch.tensor(new_lambda, dtype=torch.float)
 
    def save_model(self, actor_dir=None, critic_dir=None):
        if actor_dir is not None:
            torch.save(self.actor.state_dict(), actor_dir)
        if critic_dir is not None:
            torch.save(self.critic.state_dict(), critic_dir)
    
    def load_model(self, actor_dir=None, critic_dir=None, noise=None):
        if actor_dir is not None:
            self.actor.load_state_dict(torch.load(actor_dir, map_location=self.device))
            self.actor.to(self.device)
            if noise is not None: # add noise
                for param in self.actor.parameters():
                    param.data += torch.randn(param.data.size()) * noise
        if critic_dir is not None:
            self.critic.load_state_dict(torch.load(critic_dir, map_location=self.device))
            self.critic.to(self.device)