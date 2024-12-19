# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from models.attention import *


# hyperparameters
OUTPUT_DIM = 64
NUM_HEADS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEQ_LEN = 10
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# z score normalization
def z_score_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    return (data - mean) / std

# load data from csv file
def load_data(filename, input_name, label_name, label_size=17):
    data = pd.read_csv(filename)
    inputs = z_score_normalize(data[input_name].values)
    labels = np.clip(data[label_name].values, a_min=0, a_max=label_size-1)

    traj = deque(np.zeros((SEQ_LEN, len(input_name))), maxlen=SEQ_LEN)
    inputs_traj = []
    for i in range(len(inputs)):
        traj.append(inputs[i].tolist())
        inputs_traj.append(list(traj))
    
    return inputs_traj, labels.tolist() # shape: (batch_size, seq_len, input_size)

# create data loader
def create_data_loader(result_dir, episode_list, input_name, label_name, label_size):
    # load data
    inputs_list = []
    labels_list = []
    for i in episode_list:
        filename = os.path.join(result_dir, 'dataset/episode_' + str(i), 'learn', 'debug.csv')
        inputs, labels = load_data(filename, input_name, label_name, label_size)

        inputs_list += inputs
        labels_list += labels
    
    inputs_list = torch.tensor(np.array(inputs_list), dtype=torch.float).to(DEVICE)
    labels_list = torch.tensor(np.array(labels_list), dtype=torch.long).to(DEVICE)
    # create data loader
    data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs_list, labels_list),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    return data_loader

# train model
def train(model, train_loader, optimizer, criterion, save_dir):
    tic = time.time()
    loss_log = []
    for epoch in range(EPOCHS):
        n_batches = len(train_loader)
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if batch_idx % 500 == 0 or batch_idx == n_batches - 1:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{n_batches}, Loss: {loss.item()}')

        loss_log.append(loss.item())
        # plot loss
        plt.plot(loss_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.savefig(os.path.join(save_dir, 'channel_access_sl_gru_loss.png'))
        plt.close()
    
    time_cost = time.time() - tic
    print("Total time: {}min {:.2f}s".format(time_cost // 60, time_cost % 60))

    # save model
    torch.save(model.state_dict(), os.path.join(save_dir, 'channel_access_sl_gru.pth'))

# test model
def test(model, test_loader, criterion, save_dir):
    model.eval()
    correct = 0
    total = 0
    loss_log = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            labels_pred = model(inputs)
            loss = criterion(labels_pred, labels)
            loss_log.append(loss.item())

            y_pred = torch.argmax(labels_pred, dim=1)
            correct += (y_pred == labels).sum().item()
            total += len(labels) 

    print(f"Test accuracy: {correct / total * 100:.2f}%")

    # plot loss
    plt.plot(loss_log, 'b-', linewidth=1)
    plt.xlabel('Batch')
    plt.ylabel('Test Loss')
    plt.savefig(os.path.join(save_dir, 'channel_access_sl_gru_test_loss.png'))
    plt.close()


if __name__ == "__main__":
    input_name = ['delta_time', 'equal_backoff', 'cw', 'n_deferred', 'n_waited_slots', 
                  'n_in_queue', 'n_sending', 'n_acked', 'n_low_delay', #'n_high_delay', 'n_expired', 'n_dropped', 
                  'n_sta', 'n_neighbor_tx', 'n_neighbor_tx_data', 'n_neighbor_acked',
                  'n1', 'n5', 'n10', 'n20', 'n50', 'n100']
    label_name = 'min_neighbor_backoff'
    label_size = 2

    prefix = 'contrib/panlink'
    result_dir = os.path.join(prefix, 'result-1733811473-dataset')
    timestamp = str(int(time.time()))
    sl_dir = os.path.join(result_dir, 'sl-' + timestamp)
    if not os.path.exists(sl_dir):
        os.makedirs(sl_dir)
    train_episodes = np.arange(0, 160)
    test_episodes = np.arange(160, 200)

    # create dataloader
    train_loader = create_data_loader(result_dir, train_episodes, input_name, label_name, label_size)
    test_loader = create_data_loader(result_dir, test_episodes, input_name, label_name, label_size)

    # create model
    model = AttentionFeatureExtractor(neighbor_dim=len(input_name), self_dim=label_size, 
                                      output_dim=OUTPUT_DIM, num_heads=NUM_HEADS).to(DEVICE)

    # define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    train_attention(model, train_loader, optimizer, criterion, sl_dir)
    
    # test model
    # try:
    #     model.load_state_dict(torch.load(os.path.join(result_dir, 'channel_access_sl_gru.pth')))
    # except:
    #     print("No suitable pretrained model found.")
    test(model, test_loader, criterion, sl_dir)