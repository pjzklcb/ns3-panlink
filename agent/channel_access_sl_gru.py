# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import math
import random
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import deque
from models.gru import GRUModel


# hyperparameters
HIDDEN_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
SEQ_LEN = 10
EPOCHS = 100


# load data from csv file
def load_data(filename, input_list, output_name):
    data = pd.read_csv(filename)
    inputs = data[input_list].values
    outputs = np.clip(data[output_name].values, a_min=0, a_max=16)
    ont_hot_outputs = np.eye(17)[outputs]

    traj = deque(np.zeros((SEQ_LEN, len(input_list))), maxlen=SEQ_LEN)
    inputs_traj = []
    for i in range(len(inputs)):
        traj.append(inputs[i].tolist())
        inputs_traj.append(list(traj))
    
    return inputs_traj, ont_hot_outputs.tolist() # shape: (batch_size, seq_len, input_size)


if __name__ == "__main__":
    input_list = ['delta_time', 'equal_backoff', 'cw', 'n_deferred', 'n_waited_slots', 
                  'n_in_queue', 'n_sending', 'n_acked', 'n_low_delay', 'n_high_delay', 'n_expired', 'n_dropped', 
                  'n_sta', 'n_neighbor_tx', 'n_neighbor_tx_data', 'n_neighbor_acked',
                  'n1', 'n5', 'n10', 'n20', 'n50', 'n100']
    output_name = 'min_neighbor_backoff'

    prefix = 'contrib/panlink'
    result_dir = 'result-1733811473-dataset'
    train_episodes = np.arange(0, 100)
    test_episodes = np.arange(100, 120)

    # load data
    train_inputs = []
    train_outputs = []
    test_inputs = []
    test_outputs = []
    for i in train_episodes:
        filename = os.path.join(prefix, result_dir, 'episode_' + str(i), 'learn', 'debug.csv')
        inputs, outputs = load_data(filename, input_list, output_name)

        train_inputs += inputs
        train_outputs += outputs
    
    for i in test_episodes:
        filename = os.path.join(prefix, result_dir, 'episode_' + str(i), 'learn', 'debug.csv')
        inputs, outputs = load_data(filename, input_list, output_name)

        test_inputs += inputs
        test_outputs += outputs

    train_inputs = torch.tensor(np.array(train_inputs), dtype=torch.float)
    train_outputs = torch.tensor(np.array(train_outputs), dtype=torch.float)
    test_inputs = torch.tensor(np.array(test_inputs), dtype=torch.float)
    test_outputs = torch.tensor(np.array(test_outputs), dtype=torch.float)

    # create dataset
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_outputs)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_outputs)

    # create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # create model
    model = GRUModel(input_size=len(input_list), output_size=17, hidden_size=HIDDEN_SIZE)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train model
    loss_log = []
    for epoch in range(EPOCHS):
        total_loss = 0
        cnt = 0
        for inputs, outputs in train_loader:
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = outputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            optimizer.zero_grad()
            outputs_pred = model(inputs)
            loss = criterion(outputs_pred, outputs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cnt += 1
            if cnt % 100 == 0:
                print(cnt, loss.item())

        average_loss = total_loss / (train_inputs.size(0) // BATCH_SIZE)
        print(f'Epoch {epoch+1}/{EPOCHS}, Average loss: {average_loss}')

        loss_log.append((epoch+1, average_loss))

    # save model
    torch.save(model.state_dict(), prefix + '/agent/models/channel_access_sl_gru.pth')

    # plot loss
    plt.figure()
    plt.plot(loss_log, 'b-', linewidth=1)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.savefig(prefix + '/model/channel_access_sl_gru_loss.png')
    plt.close()

    # test model
    model.eval()
    test_loss_log = []
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        for inputs, outputs in test_loader:
            inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = outputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

            outputs_pred = model(inputs)
            loss = criterion(outputs_pred, outputs)
            print(f'Test Loss: {loss.item()}')

            test_loss_log.append(loss.item())
    
    # plot test loss
    plt.figure()
    plt.plot(test_loss_log, 'b-', linewidth=1)
    plt.legend()
    plt.xlabel('Batch idx')
    plt.ylabel('Loss')
    plt.savefig(prefix + '/model/channel_access_sl_gru_test_loss.png')
    plt.close()