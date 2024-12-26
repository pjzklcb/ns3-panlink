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
from models.neighbor_model import *
from dataset.neighbor_data_loader import *
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler


# hyperparameters
HIDDEN_DIM = 64
EMBED_DIM = 32
BATCH_SIZE = 64
EPOCHS = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__)) + "/dataset"
    train_file = os.path.join(data_dir, "data-1735132777_neighbor.h5")
    val_file = os.path.join(data_dir, "data-1735132750_neighbor.h5")
    test_file = os.path.join(data_dir, "data-1735132786_neighbor.h5")
    selected_neighbor_features = [#'role', 
                                  'n_tx_data', #'n_acked_data', 'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', #'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                                #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                                  'r_acked_data', 'tx_prob', 'tx_range_to_mean', 'tx_cv', 'tx_center_position'
                                  ]
    training_mode = False

    train_dataset = HDF5Dataset(train_file, selected_neighbor_features)
    val_dataset = HDF5Dataset(val_file, selected_neighbor_features)
    test_dataset = HDF5Dataset(test_file, selected_neighbor_features)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # create model
    enc_model = SimpleExtractor(input_dim=len(selected_neighbor_features), output_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    dec_model = SimpleDecoder(input_dim=EMBED_DIM).to(DEVICE)

    # load model
    try:
        enc_model.load_state_dict(torch.load(os.path.join(data_dir, 'neighbor_enc_model.pth')))
        dec_model.load_state_dict(torch.load(os.path.join(data_dir, 'neighbor_dec_model.pth')))
        print("Pretrained model loaded successfully.")
    except:
        print("No pretrained model found.")

    optimizer = torch.optim.Adam([{'params': enc_model.parameters()}, 
                                  {'params': dec_model.parameters()}], lr=1e-3)#, weight_decay=1e-5)
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    if training_mode:
        train_loss_log = []
        val_loss_log = []
        lr_log = []
        timestamp = int(time.time())
        for epoch in range(EPOCHS):
            train_loss = train_neighbor(enc_model, dec_model, optimizer, train_loader, device=DEVICE, clip_grad=5.0) 
            train_loss_log.append(train_loss)
            print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}")

            # evaluate
            if epoch % 10 == 0 or epoch == EPOCHS - 1:
                with torch.no_grad():
                    val_loss = eval_neighbor(enc_model, dec_model, val_loader, device=DEVICE)
                    val_loss_log.append(val_loss)
                    print(f"Epoch {epoch+1}: val loss = {val_loss:.4f}")

            # adjust learning rate
            scheduler.step()  

            # Log learning rates
            param_group = optimizer.param_groups[0]
            lr_log.append(param_group['lr'])
            print(f"Epoch {epoch+1}: Learning rate = {param_group['lr']}")

        time_cost = int(time.time()) - timestamp
        print("Total time: {}min {:.2f}s".format(time_cost // 60, time_cost % 60))

        # save model
        torch.save(enc_model.state_dict(), os.path.join(data_dir, 'neighbor_enc_model.pth'))
        torch.save(dec_model.state_dict(), os.path.join(data_dir, 'neighbor_dec_model.pth'))

        # plot loss
        plt.plot(train_loss_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(data_dir, 'neighbor_train_loss.png'))
        plt.close()

        plt.plot(val_loss_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(data_dir, 'neighbor_val_loss.png'))
        plt.close()
        
        plt.plot(lr_log, 'b-', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.savefig(os.path.join(data_dir, 'neighbor_lr.png'))
        plt.close()
    
    else:
        # test the accuracy
        accuracy = accuracy_neighbor(enc_model, dec_model, test_loader, device=DEVICE)
        print("Accuracy: {:.4f}".format(accuracy))