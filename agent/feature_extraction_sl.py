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
import torch.optim.lr_scheduler as lr_scheduler


# hyperparameters
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 4
BATCH_SIZE = 64
EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/dataset"
    train_file = os.path.join(data_dir, "data-1734607299.h5")
    val_file = os.path.join(data_dir, "data-1734591014.h5")
    selected_self_features = ['cw', 'n_defer_access', 'n_waited_slots', 'n_in_queue', 
                              'n_sending', 'n_acked', 
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

    train_dataset = HDF5Dataset(train_file, selected_self_features, selected_neighbor_features)
    val_dataset = HDF5Dataset(val_file, selected_self_features, selected_neighbor_features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # create model
    # enc_model = ImprovedAttentionExtractor(neighbor_dim=len(selected_neighbor_features),
    #                                        self_dim=len(selected_self_features),
    #                                        output_dim=EMBED_DIM,
    #                                        num_heads=NUM_HEADS).to(DEVICE)
    enc_model = TransformerExtractor(neighbor_dim=len(selected_neighbor_features),
                                     self_dim=len(selected_self_features),
                                     embed_dim=EMBED_DIM,
                                     num_heads=NUM_HEADS,
                                     num_layers=NUM_LAYERS,
                                     hidden_dim=HIDDEN_DIM).to(DEVICE)
    dec_model = SimpleDecoder(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

    # load model
    try:
        enc_model.load_state_dict(torch.load(os.path.join(data_dir, 'enc_model.pth')))
        dec_model.load_state_dict(torch.load(os.path.join(data_dir, 'dec_model.pth')))
        print("Pretrained model loaded successfully.")
    except:
        print("No pretrained model found.")

    optimizer = torch.optim.Adam([{'params': enc_model.parameters()}, 
                                  {'params': dec_model.parameters()}], lr=1e-3, weight_decay=1e-4)
    
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

    if training_mode:
        train_loss_log = []
        val_loss_log = []
        lr_log = []
        timestamp = int(time.time())
        for epoch in range(EPOCHS):
            # set models to training mode
            enc_model.train()
            dec_model.train()

            train_loss = train_attention(enc_model, dec_model, optimizer, train_loader, device=DEVICE, clip_grad=5.0) 
            train_loss_log.append(train_loss)
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}")

            # Set models to evaluation mode
            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                enc_model.eval()
                dec_model.eval()
                with torch.no_grad():
                    val_loss = eval_attention(enc_model, dec_model, val_loader, device=DEVICE)
                    val_loss_log.append(val_loss)
                    print(f"Epoch {epoch+1}: val loss = {val_loss:.4f}")

            # Adjust learning rate
            scheduler.step()  

            # Log learning rates
            param_group = optimizer.param_groups[0]
            lr_log.append(param_group['lr'])
            print(f"Epoch {epoch+1}: Learning rate = {param_group['lr']}")

        time_cost = int(time.time()) - timestamp
        print("Total time: {}min {:.2f}s".format(time_cost // 60, time_cost % 60))

        # save model
        torch.save(enc_model.state_dict(), os.path.join(data_dir, 'enc_model.pth'))
        torch.save(dec_model.state_dict(), os.path.join(data_dir, 'dec_model.pth'))

        # plot loss
        plt.plot(train_loss_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(data_dir, 'train_loss.png'))
        plt.close()

        plt.plot(val_loss_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(data_dir, 'val_loss.png'))
        plt.close()
        
        plt.plot(lr_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(data_dir, 'lr.png'))
        plt.close()
    
    else:
        # Set models to evaluation mode
        enc_model.eval()
        dec_model.eval()

        accuracy = eval_attention(enc_model, dec_model, val_loader, device=DEVICE)
        print("Accuracy: {:.4f}".format(accuracy))