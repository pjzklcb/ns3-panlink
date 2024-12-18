# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFeatureExtractor(nn.Module):
    def __init__(self, neighbor_dim, self_dim, output_dim, num_heads=1):
        super(AttentionFeatureExtractor, self).__init__()
        self.query = nn.Linear(self_dim, output_dim)
        self.key = nn.Linear(neighbor_dim, output_dim)
        self.value = nn.Linear(neighbor_dim, output_dim)
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

    def forward(self, neighbor_features, self_features, key_padding_mask=None):
        # neighbor_features: (batch_size, seq_len, neighbor_dim)
        # self_features: (batch_size, 1, self_dim)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(neighbor_features.shape[0], 
                                           neighbor_features.shape[1], dtype=torch.bool)
        Q = self.query(self_features)
        K = self.key(neighbor_features)
        V = self.value(neighbor_features)
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask) # block the True positions
        return attn_output # (batch_size, 1, output_dim)


class SimpleDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, max_seq_len=100):
        super(SimpleDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

        self.positional_encoding = self._generate_positional_encoding(max_seq_len, input_dim)

    def _generate_positional_encoding(self, max_seq_len, dim):
        """
        Use Sinusoidal Positional Encoding
        """
        pos = torch.arange(0, max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        i = torch.arange(0, dim, 2).unsqueeze(0)  # (1, dim // 2)
        div_term = torch.exp(-i / dim * torch.log(torch.tensor(10000.0)))  # 10000^(2i/dim)
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe   # (max_seq_len, dim)
    
    def forward(self, x, key_padding_mask):
        # x: (batch_size, 1, input_dim)
        # key_padding_mask: (batch_size, max_neighbors)
        batch_size = x.shape[0]
        max_n = key_padding_mask.shape[1]

        # generate pos encoding to fit the batch
        pos_encoding = self.positional_encoding[:max_n, :].to(x.device)  # (max_n, input_dim)
        pos_encoding = pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, max_n, input_dim)

        # extend latent_feature and add pos encoding
        x_expand = x.expand(-1, max_n, -1)  # (batch_size, max_n, input_dim)
        features_with_pos = x_expand + pos_encoding  # (batch_size, max_n, input_dim)

        reconstructed_features = self.fc(features_with_pos)  # (batch_size, max_n, output_dim)

        # block invalid neighbor features with 0
        key_padding_mask = key_padding_mask.unsqueeze(-1)  # (batch_size, max_n, 1)
        reconstructed_features = reconstructed_features.masked_fill(key_padding_mask, 0)

        return reconstructed_features


def chamfer_distance(reconstructed, target, key_padding_mask):
    """
    Compute Chamfer Distance between two sets of features.
    :param reconstructed: Tensor of shape (batch_size, N, feature_dim), reconstructed neighbor features
    :param target: Tensor of shape (batch_size, N, feature_dim), ground truth neighbor features
    :param key_padding_mask: Tensor of shape (batch_size, N), True indicates padding positions
    :return: Chamfer Distance loss
    """
    # Compute pairwise distances
    diff_matrix = torch.cdist(reconstructed, target, p=2)  # (batch_size, N, N)

    # Mask invalid entries in the distance matrix
    # Expand mask to match distance matrix dimensions
    mask_reconstructed = key_padding_mask.unsqueeze(2)  # (batch_size, N, 1)
    mask_target = key_padding_mask.unsqueeze(1)  # (batch_size, 1, N)

    # Apply masks: set distances involving padded points to a large value (e.g., infinity)
    diff_matrix = diff_matrix.masked_fill(mask_reconstructed, float('inf'))
    diff_matrix = diff_matrix.masked_fill(mask_target, float('inf'))

    # For each reconstructed point, find the nearest target point (ignoring padded points)
    min_dist_recon_to_target = torch.min(diff_matrix, dim=2)[0]  # (batch_size, N)

    # For each target point, find the nearest reconstructed point (ignoring padded points)
    min_dist_target_to_recon = torch.min(diff_matrix, dim=1)[0]  # (batch_size, N)

    # Mask distances for padding positions
    min_dist_recon_to_target = min_dist_recon_to_target.masked_fill(key_padding_mask, 0)
    min_dist_target_to_recon = min_dist_target_to_recon.masked_fill(key_padding_mask, 0)
    valid_samples_sum = (~key_padding_mask).sum(dim=1).float()

    # Compute Chamfer loss for each sample
    chamfer_loss = (torch.sum(min_dist_recon_to_target, dim=1) / valid_samples_sum + 
                    torch.sum(min_dist_target_to_recon, dim=1) / valid_samples_sum).sum()

    return chamfer_loss


def train_attention(enc_model, dec_model, optimizer, neighbor_features, self_features, 
                    key_padding_mask, device='cpu', clip_grad=None):
    """
    Train the attention-based encoder-decoder model for one step.
    
    :param enc_model: Encoder model to extract latent features.
    :param dec_model: Decoder model to reconstruct neighbor features.
    :param optimizer: Optimizer for training.
    :param neighbor_features: List of shape (batch_size, N, feature_dim), ground truth neighbor features.
    :param self_features: List of shape (batch_size, self_dim), self features.
    :param key_padding_mask: List of shape (batch_size, N), True for padding positions.
    :param device: Device to use ('cpu' or 'cuda').
    :param clip_grad: Optional maximum gradient norm for gradient clipping.
    :return: Loss value for the training step.
    """
    # Ensure inputs are tensors and move to the correct device
    neighbor_features = torch.tensor(neighbor_features, dtype=torch.float32, device=device)
    self_features = torch.tensor(self_features, dtype=torch.float32, device=device).unsqueeze(1)
    key_padding_mask = torch.tensor(key_padding_mask, dtype=torch.bool, device=device)

    # Set models to training mode
    enc_model.train()
    dec_model.train()

    # Zero gradients
    optimizer.zero_grad()
    
    # get the latent features
    latent_features = enc_model(neighbor_features, self_features, key_padding_mask)

    # reconstruct neighbor features
    reconstructed_neighbor_features = dec_model(latent_features, key_padding_mask)
    
    # use chamfer distance as loss function
    loss = chamfer_distance(reconstructed_neighbor_features, neighbor_features, key_padding_mask)

    # Backward pass
    loss.backward()

    # Optional: Gradient clipping
    if clip_grad is not None:
        nn.utils.clip_grad_norm_(enc_model.parameters(), max_norm=clip_grad)
        nn.utils.clip_grad_norm_(dec_model.parameters(), max_norm=clip_grad)

    # Optimization step
    optimizer.step()

    return loss.item()