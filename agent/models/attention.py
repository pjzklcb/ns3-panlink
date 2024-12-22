# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import math
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
                                           neighbor_features.shape[1], 
                                           dtype=torch.bool).to(neighbor_features.device)
        Q = self.query(self_features)
        K = self.key(neighbor_features)
        V = self.value(neighbor_features)
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask) # block the True positions
        return attn_output.squeeze(1) # (batch_size, output_dim)


class AttentionFeatureExtractor2(nn.Module):
    def __init__(self, neighbor_dim, self_dim, output_dim, num_heads=1):
        super(AttentionFeatureExtractor2, self).__init__()
        self.query = nn.Linear(self_dim, output_dim)
        self.key = nn.Linear(neighbor_dim, output_dim)
        self.value = nn.Linear(neighbor_dim, output_dim)
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)

    def forward(self, neighbor_features, self_features, key_padding_mask=None):
        # neighbor_features: (batch_size, seq_len, neighbor_dim)
        # self_features: (batch_size, 1, self_dim)
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(neighbor_features.shape[0], 
                                           neighbor_features.shape[1], 
                                           dtype=torch.bool).to(neighbor_features.device)
        Q = self.query(self_features)
        K = self.key(neighbor_features)
        V = self.value(neighbor_features)
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask) # block the True positions
        return attn_output.squeeze(1) # (batch_size, output_dim)
    

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        return F.sigmoid(self.fc2(x))


def train_attention(enc_model, dec_model, optimizer, data_loader, device='cpu', clip_grad=None):
    """
    Train the attention-based encoder-decoder model for one step.
    
    :param enc_model: Encoder model to extract latent features.
    :param dec_model: Decoder model to reconstruct neighbor features.
    :param optimizer: Optimizer for training.
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :param clip_grad: Optional maximum gradient norm for gradient clipping.
    :return: List of average chamfer distance loss values for each epoch.
    """
    epoch_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for batch_idx, (self_features, neighbor_features, key_padding_mask, min_neighbor_backoff) in enumerate(data_loader):
        # Ensure inputs are tensors and move to the correct device
        self_features = self_features.to(device)
        neighbor_features = neighbor_features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        min_neighbor_backoff = min_neighbor_backoff.to(device)

        # Convert min_neighbor_backoff to binary
        min_neighbor_backoff[min_neighbor_backoff > 0] = 1.0

        # Zero gradients
        optimizer.zero_grad()
        
        # get the embeddings
        embeddings = enc_model(neighbor_features, self_features, key_padding_mask)

        # do the classification
        y_pred = dec_model(embeddings)
        
        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, min_neighbor_backoff.unsqueeze(1))

        # Backward pass
        loss.backward()

        # Optional: Gradient clipping
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(enc_model.parameters(), max_norm=clip_grad)
            nn.utils.clip_grad_norm_(dec_model.parameters(), max_norm=clip_grad)

        # Optimization step
        optimizer.step()

        epoch_loss += loss.item()
    
        if batch_idx % 500 == 0 or batch_idx == num_batches - 1:
            print(f"Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.2f}")
    
    return epoch_loss / num_batches


def eval_attention(enc_model, dec_model, data_loader, device='cpu'):
    """
    Evaluate the attention-based encoder-decoder model.
    
    :param enc_model: Encoder model to extract latent features.
    :param dec_model: Decoder model to reconstruct neighbor features.
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :return: the average loss over the test set
    """
    total_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for batch_idx, (self_features, neighbor_features, key_padding_mask, min_neighbor_backoff) in enumerate(data_loader):
        # Ensure inputs are tensors and move to the correct device
        self_features = self_features.to(device)
        neighbor_features = neighbor_features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        min_neighbor_backoff = min_neighbor_backoff.to(device)

        # Convert min_neighbor_backoff to binary
        min_neighbor_backoff[min_neighbor_backoff > 0] = 1.0

        # get the embeddings
        embeddings = enc_model(neighbor_features, self_features, key_padding_mask)

        # do the classification
        y_pred = dec_model(embeddings)

        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, min_neighbor_backoff.unsqueeze(1))

        total_loss += loss.item()

        if batch_idx % 500 == 0 or batch_idx == num_batches - 1:
                print(f"Batch {batch_idx+1}/{num_batches}")

    return total_loss / num_batches


def accuracy_attention(enc_model, dec_model, data_loader, device='cpu'):
    """
    Test the classification accuracy attention-based encoder-decoder model.
    
    :param enc_model: Encoder model to extract latent features.
    :param dec_model: Decoder model to reconstruct neighbor features.
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :return: the accuracy over the test set
    """
    accuracy = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for batch_idx, (self_features, neighbor_features, key_padding_mask, min_neighbor_backoff) in enumerate(data_loader):
        # Ensure inputs are tensors and move to the correct device
        self_features = self_features.to(device)
        neighbor_features = neighbor_features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        min_neighbor_backoff = min_neighbor_backoff.to(device)

        # Convert min_neighbor_backoff to binary
        min_neighbor_backoff[min_neighbor_backoff > 0] = 1.0

        # get the embeddings
        embeddings = enc_model(neighbor_features, self_features, key_padding_mask)

        # do the classification
        y_pred = dec_model(embeddings)

        # judge if do the correct classification
        y_pred = (y_pred > 0.5).int().squeeze()

        accuracy += (y_pred == min_neighbor_backoff).sum().item()

        if batch_idx % 500 == 0 or batch_idx == num_batches - 1:
                print(f"Batch {batch_idx+1}/{num_batches}")

    return accuracy / len(data_loader.dataset)