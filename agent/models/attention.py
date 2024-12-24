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


class ImprovedAttentionExtractor(nn.Module):
    def __init__(self, neighbor_dim, self_dim, output_dim, num_heads=1, pooling="max"):
        super(ImprovedAttentionExtractor, self).__init__()
        self.self_query = nn.Linear(self_dim, output_dim)
        self.query = nn.Linear(neighbor_dim, output_dim)
        self.key = nn.Linear(neighbor_dim, output_dim)
        self.value = nn.Linear(neighbor_dim, output_dim)
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
        
        # Pooling method: mean or max
        assert pooling in ["mean", "max"], "Pooling must be 'mean' or 'max'"
        self.pooling = pooling

    def forward(self, neighbor_features, self_features, key_padding_mask=None):
        # neighbor_features: (batch_size, seq_len, neighbor_dim)
        # self_features: (batch_size, 1, self_dim)

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(neighbor_features.shape[0], 
                                           neighbor_features.shape[1], 
                                           dtype=torch.bool).to(neighbor_features.device)

        # Linear transformations
        self_Q = self.self_query(self_features)  # (batch_size, 1, output_dim)
        Q = self.query(neighbor_features)        # (batch_size, seq_len, output_dim)
        K = self.key(neighbor_features)          # (batch_size, seq_len, output_dim)
        V = self.value(neighbor_features)        # (batch_size, seq_len, output_dim)

        # Attention computations
        self_attn, _ = self.mha(self_Q, K, V, key_padding_mask=key_padding_mask)  # (batch_size, 1, output_dim)
        neighbor_attn, _ = self.mha(Q, K, V, key_padding_mask=key_padding_mask)  # (batch_size, seq_len, output_dim)

        # Pooling neighbor_attn
        if self.pooling == "mean":
            pooled_neighbor_attn = neighbor_attn.masked_fill(key_padding_mask.unsqueeze(-1), 0).mean(dim=1)  # (batch_size, output_dim)
        elif self.pooling == "max":
            pooled_neighbor_attn = neighbor_attn.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]  # (batch_size, output_dim)

        # Concatenate self_attn and pooled_neighbor_attn
        attn_output = torch.cat([self_attn.squeeze(1), pooled_neighbor_attn], dim=-1)  # (batch_size, 2 * output_dim)
        return attn_output


class FullAttentionExtractor(nn.Module):
    def __init__(self, neighbor_dim, self_dim, output_dim, num_heads=1, pooling="max"):
        super(FullAttentionExtractor, self).__init__()
        self.self_query = nn.Linear(self_dim, output_dim)
        self.self_key = nn.Linear(self_dim, output_dim)
        self.self_value = nn.Linear(self_dim, output_dim)
        self.query = nn.Linear(neighbor_dim, output_dim)
        self.key = nn.Linear(neighbor_dim, output_dim)
        self.value = nn.Linear(neighbor_dim, output_dim)
        self.mha = nn.MultiheadAttention(embed_dim=output_dim, num_heads=num_heads, batch_first=True)
        
        # Pooling method: mean or max
        assert pooling in ["mean", "max"], "Pooling must be 'mean' or 'max'"
        self.pooling = pooling

    def forward(self, neighbor_features, self_features, key_padding_mask=None):
        # neighbor_features: (batch_size, seq_len, neighbor_dim)
        # self_features: (batch_size, 1, self_dim)

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(neighbor_features.shape[0], 
                                           neighbor_features.shape[1], 
                                           dtype=torch.bool).to(neighbor_features.device)

        # Linear transformations
        self_Q = self.self_query(self_features)  # (batch_size, 1, output_dim)
        self_K = self.self_key(self_features)    # (batch_size, 1, output_dim)
        self_V = self.self_value(self_features)  # (batch_size, 1, output_dim)
        neighbor_Q = self.query(neighbor_features)        # (batch_size, seq_len, output_dim)
        neighbor_K = self.key(neighbor_features)          # (batch_size, seq_len, output_dim)
        neighbor_V = self.value(neighbor_features)        # (batch_size, seq_len, output_dim)
        Q = torch.cat([self_Q, neighbor_Q], dim=1)        # (batch_size, seq_len + 1, output_dim)
        K = torch.cat([self_K, neighbor_K], dim=1)        # (batch_size, seq_len + 1, output_dim)
        V = torch.cat([self_V, neighbor_V], dim=1)        # (batch_size, seq_len + 1, output_dim)
        M = torch.cat([torch.zeros(key_padding_mask.shape[0], 1, dtype=torch.bool).to(key_padding_mask.device), 
                       key_padding_mask], dim=1)

        # Attention computations
        attn_output, _ = self.mha(Q, K, V, key_padding_mask=M)  # (batch_size, seq_len + 1, output_dim)

        self_attn = attn_output[:, :1, :]  # (batch_size, 1, output_dim)
        neighbor_attn = attn_output[:, 1:, :]  # (batch_size, seq_len, output_dim)

        # Pooling neighbor_attn
        if self.pooling == "mean":
            pooled_neighbor_attn = neighbor_attn.masked_fill(key_padding_mask.unsqueeze(-1), 0).mean(dim=1)  # (batch_size, output_dim)
        elif self.pooling == "max":
            pooled_neighbor_attn = neighbor_attn.masked_fill(key_padding_mask.unsqueeze(-1), float("-inf")).max(dim=1)[0]  # (batch_size, output_dim)

        # Concatenate self_attn and pooled_neighbor_attn
        attn_output = torch.cat([self_attn.squeeze(1), pooled_neighbor_attn], dim=-1)  # (batch_size, 2 * output_dim)
        return attn_output
    

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.sigmoid(self.fc2(x))


class TransformerExtractor(nn.Module):
    def __init__(self, neighbor_dim, self_dim, embed_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerExtractor, self).__init__()
        
        # 输入特征嵌入
        self.neighbor_embed = nn.Linear(neighbor_dim, embed_dim)
        self.self_embed = nn.Linear(self_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, neighbor_features, self_feature, key_padding_mask=None):
        """
        neighbor_features: (batch_size, N, neighbor_dim)
        self_feature: (batch_size, 1, self_dim)
        key_padding_mask: (batch_size, N), where True indicates padding positions
        """
        batch_size, num_neighbors, _ = neighbor_features.shape

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(batch_size, num_neighbors, 
                                           dtype=torch.bool,
                                           device=neighbor_features.device)
            
        # get embeddings
        neighbor_embedded = self.neighbor_embed(neighbor_features)  # (batch_size, N, embed_dim)
        self_embedded = self.self_embed(self_feature)  # (batch_size, 1, embed_dim)
        
        # apply dropout
        neighbor_embedded = self.dropout(neighbor_embedded)
        self_embedded = self.dropout(self_embedded)
        embeddings = torch.cat([self_embedded, neighbor_embedded], dim=1) # (batch_size, N+1, embed_dim)
        combined_mask = torch.cat(
            [torch.zeros(batch_size, 1, 
                         dtype=torch.bool, 
                         device=key_padding_mask.device), 
             key_padding_mask], 
            dim=1
        )

        # extract features using encoder
        encoded_output = self.encoder(embeddings, 
                                      src_key_padding_mask=combined_mask)  # (N, batch_size, embed_dim)
        
        return encoded_output[:, 0, :]  # (batch_size, embed_dim)


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
    # set to training mode
    enc_model.train()
    dec_model.train()
    
    epoch_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for batch_idx, (self_features, neighbor_features, key_padding_mask, min_neighbor_backoff) in enumerate(data_loader):
        # Ensure inputs are tensors and move to the correct device
        self_features = self_features.to(device)
        neighbor_features = neighbor_features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        min_neighbor_backoff = min_neighbor_backoff.to(device)

        # Convert min_neighbor_backoff to soft label
        y_true = torch.exp(-0.5*min_neighbor_backoff)

        # Zero gradients
        optimizer.zero_grad()
        
        # get the embeddings
        embeddings = enc_model(neighbor_features, self_features, key_padding_mask)

        # do the classification
        y_pred = dec_model(embeddings)
        
        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true.unsqueeze(1))

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
    # set to evaluation mode
    enc_model.eval()
    dec_model.eval()
    
    total_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for self_features, neighbor_features, key_padding_mask, min_neighbor_backoff in data_loader:
        # Ensure inputs are tensors and move to the correct device
        self_features = self_features.to(device)
        neighbor_features = neighbor_features.to(device)
        key_padding_mask = key_padding_mask.to(device)
        min_neighbor_backoff = min_neighbor_backoff.to(device)

        # Convert min_neighbor_backoff to soft label
        y_true = torch.exp(-0.5*min_neighbor_backoff)

        # get the embeddings
        embeddings = enc_model(neighbor_features, self_features, key_padding_mask)

        # do the classification
        y_pred = dec_model(embeddings)

        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true.unsqueeze(1))

        total_loss += loss.item()

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
    # set to evaluation mode
    enc_model.eval()
    dec_model.eval()

    accuracy = 0
    for self_features, neighbor_features, key_padding_mask, min_neighbor_backoff in data_loader:
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
        y_pred = (y_pred < 0.5).int().squeeze()

        accuracy += (y_pred == min_neighbor_backoff).sum().item()

    return accuracy / len(data_loader.dataset)