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
from dataset.data_loader import *
from torch.utils.data import DataLoader


# hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
NUM_HEADS = 4
BATCH_SIZE = 32
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/dataset"
    file_path =  os.path.join(data_dir, "data-1734690536.h5")
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
    training_mode = True

    dataset = HDF5Dataset(file_path, selected_self_features, selected_neighbor_features)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # create model
    enc_model = AttentionFeatureExtractor(neighbor_dim=len(selected_neighbor_features),
                                          self_dim=len(selected_self_features),
                                          output_dim=EMBEDDING_DIM,
                                          num_heads=NUM_HEADS).to(DEVICE)
    dec_model = SimpleDecoder(input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    # load model
    try:
        enc_model.load_state_dict(torch.load(os.path.join(data_dir, 'attn_model1.pth')))
        dec_model.load_state_dict(torch.load(os.path.join(data_dir, 'dec_model1.pth')))
        print("Pretrained model loaded successfully.")
    except:
        print("No pretrained model found.")

    optimizer = torch.optim.Adam([{'params': enc_model.parameters(), 'lr': 1e-4}, 
                                  {'params': dec_model.parameters(), 'lr': 1e-3}])
    
    if training_mode:
        # set models to training mode
        enc_model.train()
        dec_model.train()

        timestamp = int(time.time())
        loss_log = []
        for epoch in range(EPOCHS):
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            loss = train_attention(enc_model, dec_model, optimizer, data_loader, device=DEVICE, clip_grad=1.0) 
            loss_log.append(loss)
            print(f"Epoch {epoch+1}: loss = {loss:.4f}")

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
    
    else:
        # Set models to evaluation mode
        enc_model.eval()
        dec_model.eval()

        accuracy = eval_attention(enc_model, dec_model, data_loader, device=DEVICE)
        print("Accuracy: {:.4f}".format(accuracy))