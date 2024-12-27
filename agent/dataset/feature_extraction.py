# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy
from scipy.stats import chi2_contingency
from data_loader import *


def calculate_mutual_info(x, y):
    h_x = entropy(np.unique(x, return_counts=True)[1])
    h_y = entropy(np.unique(y, return_counts=True)[1])
    h_xy = entropy(np.histogram2d(x, y, bins=[len(np.unique(x)), len(np.unique(y))])[0].flatten())

    mi = h_x + h_y - h_xy
    return mi


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735182796_trans.h5"
    selected_self_features = [#'delta_time', 'cw', 
                                'n_defer_access', 'n_waited_slots', 
                            #   'n_in_queue', 'n_sending', 'n_acked', 
                            #   'n_tx_data', 'n_acked_data', #'last_tx_delay', 
                            #   'last_txok_delay', 'chan_util', 
                                'n_rx_fail', #'idle_ratio',
                            #   'app_interval_mean', 'app_interval_std', 'app_interval_min', 'app_interval_max',
                            #   'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                            #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                            #   'tx_prob', 'tx_range_to_mean', 'tx_cv', 'tx_center_position'
                              ]
    selected_neighbor_features = [#'role', 
                                #   'n_tx_data', 'n_acked_data', #'last_tx_delay', 
                                #   'last_txok_delay', 'chan_util', #'mean_rssi', 
                                #   'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                                #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                                  'tx_prob', #'tx_range_to_mean', 'tx_cv', 'tx_center_position'
                                  'backoff'
                                  ]
    
    dataset = HDF5Dataset(file_path, selected_self_features, selected_neighbor_features)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    # # test data loader
    # for batch_idx, (self_features, padded_neighbors, key_padding_mask, min_neighbor_backoffs) in enumerate(data_loader):
    #     if batch_idx >= 1:
    #         break
    #     print(f"Batch {batch_idx + 1}:")
    #     print("Self Features Shape:", self_features.shape)
    #     print("Padded Neighbors Shape:", padded_neighbors.shape)
    #     print("Key Padding Mask Shape:", key_padding_mask.shape)
    #     print("Min Neighbor Backoffs Shape:", min_neighbor_backoffs.shape)
    
    # analyze dataset
    self_features = np.zeros((len(data_loader.dataset), len(selected_self_features)))
    # neighbor_features = np.zeros((len(data_loader.dataset), len(selected_neighbor_features)))
    tx_prob = np.zeros((len(data_loader.dataset), 20))
    backoff = np.zeros((len(data_loader.dataset), 20))
    labels = np.zeros((len(data_loader.dataset), 1))
    cnt = 0
    for feature, n_feature, key_padding_mask, label in data_loader:
        batch_size = feature.shape[0]
        self_features[cnt:cnt + batch_size] = feature.squeeze(1).numpy()
        tx_prob[cnt:cnt + batch_size] = n_feature[:, :, 0].view(batch_size, -1).numpy()
        backoff[cnt:cnt + batch_size] = n_feature[:, :, 1].view(batch_size, -1).numpy()
        # neighbor_features[cnt:cnt + batch_size] = tx_prob.numpy()
        labels[cnt:cnt + batch_size] = label.unsqueeze(1).numpy()
        cnt += batch_size

    s_df = pd.DataFrame(self_features, columns=selected_self_features)
    # n_df = pd.DataFrame(neighbor_features, columns=list(range(20)))
    tx_df = pd.DataFrame(tx_prob, columns=list(range(20)))
    bo_df = pd.DataFrame(backoff, columns=list(range(20)))
    labels = pd.DataFrame(labels, columns=['min_neighbor_backoff'])#.clip(lower=0, upper=1)

    x = pd.concat([s_df, tx_df, bo_df], axis=1)
    x.to_csv('contrib/panlink/agent/output.csv', index=True)
    # calculate spearman correlation
    # for key in selected_self_features:
    #     metric = labels['min_neighbor_backoff'].corr(s_df[key], method='spearman')
    #     print(f"{key:>20}  {metric:.4f}")
    # for key in selected_neighbor_features:
    #     metric = labels['min_neighbor_backoff'].corr(n_df[key], method='spearman')
    #     print(f"{key:>20}  {metric:.4f}")
    # metric = labels['min_neighbor_backoff'].corr(labels['min_neighbor_backoff'], method='spearman')
    # print(f"{'min_neighbor_backoff':>20}  {metric:.4f}")

    # calculate chi square test
    # for key in selected_self_features:
    #     chi2, p, dof, _ = chi2_contingency(pd.crosstab(s_df[key], labels['min_neighbor_backoff']))
    #     print(f"{key:>20}: chi^2 = {chi2:.2f}, p = {p:.2f}, dof = {dof:.2f}")
    # for key in selected_neighbor_features:
    #     chi2, p, dof, _ = chi2_contingency(pd.crosstab(n_df[key], labels['min_neighbor_backoff']))
    #     print(f"{key:>20}: chi^2 = {chi2:.2f}, p = {p:.2f}, dof = {dof:.2f}")

    # x = n_df['tx_prob']
    # # x = s_df['n_rx_drop']
    # y = labels['min_neighbor_backoff']
    # plt.hist(x[y==0], bins=20, alpha=0.5, label='0', density=True)
    # plt.hist(x[y==1], bins=20, alpha=0.5, label='1', density=True)
    # plt.legend()
    # plt.show()