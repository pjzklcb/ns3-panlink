# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.attention import *
from dataset.data_loader import H5DataLoader


# hyperparameters
LATENT_DIM = 64
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/dataset"
    file_path =  os.path.join(data_dir, "data-1734528339.h5")
    selected_self_features = [#'cw', 'n_defer_access', 'n_waited_slots', 'n_in_queue', 
                              #'n_sending', 'n_acked', 
                              'n_tx_data', 'n_acked_data', 
                              'last_tx_delay', 'last_txok_delay', 'chan_util', 
                              'app_interval_mean', 'app_interval_std', 'app_interval_min', 
                              'app_interval_max', 'tx_interval_mean', 'tx_interval_std', 
                              'tx_interval_min', 'tx_interval_max', 'tx_duration_mean', 
                              'tx_duration_std', 'tx_duration_min', 'tx_duration_max']
    selected_neighbor_features = ['role', 'n_tx_data', 'n_acked_data', 'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', 'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', 'tx_interval_min', 
                                  'tx_interval_max', 'tx_duration_mean', 'tx_duration_std', 
                                  'tx_duration_min', 'tx_duration_max']

    data_loader = H5DataLoader(file_path, selected_self_features, selected_neighbor_features, 
                               batch_size=BATCH_SIZE)

    # create model
    enc_model = AttentionFeatureExtractor(neighbor_dim=len(selected_neighbor_features),
                                          self_dim=len(selected_self_features),
                                          output_dim=LATENT_DIM,
                                          num_heads=NUM_HEADS).to(DEVICE)
    dec_model = SimpleDecoder(input_dim=LATENT_DIM, 
                              output_dim=len(selected_neighbor_features)).to(DEVICE)

    optimizer = torch.optim.Adam([{'params': enc_model.parameters(), 'lr': 1e-4}, 
                                  {'params': dec_model.parameters(), 'lr': 1e-3}])
    
    timestamp = int(time.time())
    loss_log = []
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch_idx, (batch_self_features, batch_neighbor_features, key_padding_mask) in enumerate(data_loader):
            loss = train_attention(enc_model, dec_model, optimizer, batch_neighbor_features, 
                                   batch_self_features, key_padding_mask, device=DEVICE, clip_grad=1.0)
            epoch_loss += loss
            if batch_idx % 500 == 0 or batch_idx == data_loader.num_batches - 1:
                print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx+1}/{data_loader.num_batches}, Loss: {loss}')

        loss_log.append(epoch_loss / data_loader.num_batches)
    
    time_cost = int(time.time()) - timestamp
    print("Total time: {}min {:.2f}s".format(time_cost // 60, time_cost % 60))
    # plot loss
    plt.plot(loss_log, 'b-', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(data_dir, 'attn_loss.png'))
    plt.close()

    # save model
    torch.save(enc_model.state_dict(), os.path.join(data_dir, 'attn_model.pth'))
    torch.save(dec_model.state_dict(), os.path.join(data_dir, 'dec_model.pth'))