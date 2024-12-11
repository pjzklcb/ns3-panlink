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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.model_library import DNNModel


# hyperparameters
HIDDEN_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEQ_LEN = 10
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load data from csv file
def load_data(result_dir, episode_list, features_name, label_name, label_size):
    # load data
    features_list = []
    labels_list = []
    for i in episode_list:
        filename = os.path.join(result_dir, 'dataset/episode_' + str(i), 'learn', 'debug.csv')
        data = pd.read_csv(filename)
        features_list.append(data[features_name])
        labels_list.append(data[label_name])

    features = pd.concat(features_list, axis=0)
    labels = pd.concat(labels_list, axis=0).clip(lower=0, upper=label_size-1)
    return features, labels

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
    features_name = ['delta_time', 'equal_backoff', 'cw', 'n_deferred', 'n_waited_slots', 
                  'n_in_queue', 'n_sending', 'n_acked', 'n_low_delay', 'n_high_delay', 'n_expired', 'n_dropped', 
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

    episode_list = np.arange(0, 200)

    # load data
    features, labels = load_data(result_dir, episode_list, features_name, label_name, label_size)
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # divide train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # create data loader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(train_features, dtype=torch.float).to(DEVICE), 
                                       torch.tensor(train_labels.values, dtype=torch.long).to(DEVICE)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(test_features, dtype=torch.float).to(DEVICE), 
                                       torch.tensor(test_labels.values, dtype=torch.long).to(DEVICE)),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # create model
    model = DNNModel(input_size=len(features_name), output_size=label_size, hidden_size=HIDDEN_SIZE)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    train(model, train_loader, optimizer, criterion, sl_dir)
    
    # test model
    # try:
    #     model.load_state_dict(torch.load(os.path.join(result_dir, 'channel_access_sl_gru.pth')))
    # except:
    #     print("No suitable pretrained model found.")
    test(model, test_loader, criterion, sl_dir)