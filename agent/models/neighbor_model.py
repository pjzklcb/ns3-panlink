# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
    

class SimpleExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(SimpleExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))
    

class SimpleDecoder(nn.Module):
    def __init__(self, input_dim):
        super(SimpleDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return F.sigmoid(self.fc(x))


def train_neighbor(enc_model, dec_model, optimizer, data_loader, device='cpu', clip_grad=None):
    """
    Train the model for one epoch
    
    :param model: Model to train
    :param optimizer: Optimizer for training.
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :param clip_grad: Optional maximum gradient norm for gradient clipping.
    :return: List of average loss values for each epoch.
    """
    # set to training mode
    enc_model.train()
    dec_model.train()
    
    epoch_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for batch_idx, (neighbor_features, backoffs) in enumerate(data_loader):
        # Ensure inputs are tensors and move to the correct device
        neighbor_features = neighbor_features.to(device)
        backoffs = backoffs.to(device)

        # Convert backoff to binary label
        y_true = (backoffs == 1024).float()

        # do the classification
        embedings = enc_model(neighbor_features)
        y_pred = dec_model(embedings)
        
        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true.unsqueeze(1))

        # Backward pass
        optimizer.zero_grad()
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


def eval_neighbor(enc_model, dec_model, data_loader, device='cpu'):
    """
    Evaluate the model.
    
    :param model: Model to evaluate
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :return: the average loss over the test set
    """
    # set to evaluation mode
    enc_model.eval()
    dec_model.eval()
    
    total_loss = 0
    num_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)
    for neighbor_features, backoffs in data_loader:
        # Ensure inputs are tensors and move to the correct device
        neighbor_features = neighbor_features.to(device)
        backoffs = backoffs.to(device)

        # Convert backoff to binary label
        y_true = (backoffs == 1024).float()
        
        # do the classification
        embedings = enc_model(neighbor_features)
        y_pred = dec_model(embedings)
        
        # use BCE as loss function
        criterion = nn.BCELoss()
        loss = criterion(y_pred, y_true.unsqueeze(1))

        total_loss += loss.item()

    return total_loss / num_batches


def accuracy_neighbor(enc_model, dec_model, data_loader, device='cpu'):
    """
    Test the classification accuracy of the model.
    
    :param enc_model: 
    :param data_loader: Data loader to generate batches.
    :param device: Device to use ('cpu' or 'cuda').
    :return: the accuracy over the test set
    """
    # set to evaluation mode
    enc_model.eval()
    dec_model.eval()

    accuracy = 0
    for neighbor_features, backoffs in data_loader:
        # Ensure inputs are tensors and move to the correct device
        neighbor_features = neighbor_features.to(device)
        backoffs = backoffs.to(device)

        # Convert backoff to binary label
        y_true = (backoffs == 1024).float()
        
        # do the classification
        embedings = enc_model(neighbor_features)
        y_pred = dec_model(embedings)

        # judge if do the correct classification
        y_pred = (y_pred > 0.5).int().squeeze()

        accuracy += (y_pred == y_true).sum().item()

    return accuracy / len(data_loader.dataset)