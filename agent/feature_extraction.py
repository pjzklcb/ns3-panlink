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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from scipy.stats import chi2_contingency


def calculate_mutual_info(x, y):
    h_x = entropy(np.unique(x, return_counts=True)[1])
    h_y = entropy(np.unique(y, return_counts=True)[1])
    h_xy = entropy(np.histogram2d(x, y, bins=[len(np.unique(x)), len(np.unique(y))])[0].flatten())

    mi = h_x + h_y - h_xy
    return mi

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


if __name__ == "__main__":
    features_name = ['delta_time', 'equal_backoff', 'cw', 'n_deferred', 'n_waited_slots', 
                  'n_in_queue', 'n_sending', 'n_acked', 'n_low_delay', 'n_high_delay', 'n_expired', 'n_dropped', 
                  'n_sta', 'n_neighbor_tx', 'n_neighbor_tx_data', 'n_neighbor_acked',
                  'n1', 'n5', 'n10', 'n20', 'n50', 'n100', 'nn']
    label_name = 'min_neighbor_backoff'
    label_size = 2

    prefix = 'contrib/panlink'
    result_dir = os.path.join(prefix, 'result-1733811473-dataset')
    episode_list = np.arange(0, 200)

    # create dataloader
    features, labels = load_data(result_dir, episode_list, features_name, label_name, label_size)
    
    # scaler = StandardScaler()
    # features = scaler.fit_transform(features)

    # calculate spearman correlation
    for key in features_name:
        metric = labels.corr(features[key], method='spearman')
        print(f"{key:>20}  {metric:.4f}")
    metric = labels.corr(labels, method='spearman')
    print(f"{label_name:>20}  {metric:.4f}")

    # calculate chi square test
    for key in features_name:
        chi2, p, dof, _ = chi2_contingency(pd.crosstab(features[key], labels.values))
        print(f"{key:>20}: chi^2 = {chi2:.2f}, p = {p:.2f}, dof = {dof:.2f}")