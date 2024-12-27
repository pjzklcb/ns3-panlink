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
from neighbor_data_loader import *


def calculate_mutual_info(x, y):
    h_x = entropy(np.unique(x, return_counts=True)[1])
    h_y = entropy(np.unique(y, return_counts=True)[1])
    h_xy = entropy(np.histogram2d(x, y, bins=[len(np.unique(x)), len(np.unique(y))])[0].flatten())

    mi = h_x + h_y - h_xy
    return mi


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735182796_neighbor.h5"
    selected_neighbor_features = [#'role', 
                                  'n_tx_data', #'n_acked_data', 'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', #'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                                #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                                  'n_rx_fail', 'idle_ratio',
                                  'r_acked_data', 'tx_prob', 'tx_range_to_mean', 'tx_cv', 'tx_center_position',
                                  'mean_idle_slots'
                                  ]
    
    dataset = HDF5Dataset(file_path, selected_neighbor_features)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # analyze dataset
    neighbor_features = np.zeros((len(data_loader.dataset), len(selected_neighbor_features)))
    labels = np.zeros((len(data_loader.dataset), 1))
    cnt = 0
    for n_feature, label in data_loader:
        batch_size = n_feature.shape[0]
        neighbor_features[cnt:cnt + batch_size] = n_feature.numpy()
        labels[cnt:cnt + batch_size] = label.unsqueeze(1).numpy()
        cnt += batch_size

    n_df = pd.DataFrame(neighbor_features, columns=selected_neighbor_features)
    labels = pd.DataFrame(labels, columns=['backoff'])
    filtered_indices = labels['backoff'] != 1024
    n_df = n_df[filtered_indices]
    labels = labels[filtered_indices]

    # calculate mutual information
    # for key in selected_neighbor_features:
    #     metric = calculate_mutual_info(n_df[key], labels['backoff'])
    #     print(f"{key:>20}  {metric:.4f}")
    # metric = calculate_mutual_info(labels['backoff'], labels['backoff'])
    # print(f"{'backoff':>20}  {metric:.4f}")

    # calculate spearman correlation
    for key in selected_neighbor_features:
        metric = labels['backoff'].corr(n_df[key], method='spearman')
        print(f"{key:>20}  {metric:.4f}")
    metric = labels['backoff'].corr(labels['backoff'], method='spearman')
    print(f"{'backoff':>20}  {metric:.4f}")

    # calculate chi square test
    for key in selected_neighbor_features:
        chi2, p, dof, _ = chi2_contingency(pd.crosstab(n_df[key], labels['backoff']))
        print(f"{key:>20}: chi^2 = {chi2:.2f}, p = {p:.2f}, dof = {dof:.2f}")

    x = n_df['tx_prob']
    y = labels['backoff']
    plt.hist(x[y==0], bins=20, alpha=0.5, label='0', density=True)
    plt.hist(x[y!=0], bins=20, alpha=0.5, label='1', density=True)
    plt.legend()
    plt.show()