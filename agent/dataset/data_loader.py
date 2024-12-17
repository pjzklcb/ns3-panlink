# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

class H5DataLoader:
    def __init__(self, file_path, selected_self_features, selected_neighbor_features, batch_size, shuffle=True):
        """
        Initialize data loader

        Parameters:
        - file_path (str): HDF5 file path.
        - selected_self_features (list of str): self feature names to be selected
        - selected_neighbor_features (list of str): neighbor feature names to be selected
        - batch_size (int): batch size
        - shuffle (bool): whether to shuffle the data
        """
        self.file_path = file_path
        self.selected_self_features = selected_self_features
        self.selected_neighbor_features = selected_neighbor_features
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = self._load_data()
        self.num_batches = len(self.data) // batch_size

    def _load_data(self):
        """
        read data from HDF5 and 
        """
        data = []

        with h5py.File(self.file_path, 'r') as f:
            self_feature_names = f["step_0/self_features"].attrs.get("feature_names")
            neighbor_feature_names = f['step_0/neighbor_features'].attrs.get("feature_names")

            selected_self_indices = [
                i for i, name in enumerate(self_feature_names)
                if name in self.selected_self_features
            ]
            selected_neighbor_indices = [
                i for i, name in enumerate(neighbor_feature_names)
                if name in self.selected_neighbor_features
            ]

            for step_key in f.keys():
                step_data = f[step_key]
                self_features_data = step_data['self_features'][:]
                neighbor_features_data = step_data['neighbor_features'][:]

                selected_self_features_data = self_features_data[selected_self_indices]
                selected_neighbor_features_data = neighbor_features_data[:, selected_neighbor_indices]

                data.append({
                    'self_features': selected_self_features_data,
                    'neighbor_features': selected_neighbor_features_data
                })

        if self.shuffle:
            random.shuffle(data)  # random shuffle

        return data

    def _normalize_batch(self, batch_self_features, batch_neighbor_features):
        """
        normalize each batch
        """
        self_scaler = StandardScaler()
        normalized_self_features = self_scaler.fit_transform(np.array(batch_self_features))

        all_neighbor_features = []
        original_neighbor_sizes = []
        
        # stack each sample's neighbor features
        for i in range(len(batch_neighbor_features)):
            N = len(batch_neighbor_features[i])  # number of neighbors of current sample
            all_neighbor_features.extend(batch_neighbor_features[i])
            original_neighbor_sizes.append(N) 

        # normlize all neighbor features
        all_neighbor_features = np.array(all_neighbor_features)
        neighbor_scaler = StandardScaler()
        normalized_neighbor_features = neighbor_scaler.fit_transform(all_neighbor_features)

        # split normalized neighbor features
        split_neighbor_features = []
        start_idx = 0
        for size in original_neighbor_sizes:
            end_idx = start_idx + size
            split_neighbor_features.append(normalized_neighbor_features[start_idx:end_idx])
            start_idx = end_idx

        return normalized_self_features, split_neighbor_features

    def _padding_and_masking(self, batch_neighbor_features, max_neighbors):
        """
        padding to have same dim and masking the invalid neighbor features
        """
        mask = []
        padded_neighbors = []
        for neighbors in batch_neighbor_features:
            N = len(neighbors)
            padded_neighbors.append(np.pad(neighbors, ((0, max_neighbors - N), (0, 0)), 
                                           mode='constant', constant_values=0))
            current_mask = np.ones(max_neighbors)  # intialize mask
            current_mask[N:] = 0  # padding invalid part
            mask.append(current_mask)
        return np.array(padded_neighbors), np.array(mask)
    
    def __iter__(self):
        """
        generate batches
        """
        batch_self_features = []
        batch_neighbor_features = []

        for i, sample in enumerate(self.data):
            batch_self_features.append(sample['self_features'])
            batch_neighbor_features.append(sample['neighbor_features'])

            if len(batch_self_features) == self.batch_size or i == len(self.data) - 1:
                # normalize current batch
                normalized_self_features, normalized_neighbor_features = self._normalize_batch(
                    batch_self_features, batch_neighbor_features
                )

                # get the maximum number of neighbors in the batch
                max_neighbors = max(len(neighbor) for neighbor in batch_neighbor_features)

                padded_neighbors, mask = self._padding_and_masking(normalized_neighbor_features, 
                                                                   max_neighbors)

                yield normalized_self_features, padded_neighbors, mask

                # empty current batch
                batch_self_features = []
                batch_neighbor_features = []


if __name__ == '__main__':
    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1734417926.h5"
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
    batch_size = 32

    data_loader = H5DataLoader(file_path, selected_self_features, selected_neighbor_features, batch_size)

    print(selected_self_features)
    for i, (batch_self_features, batch_neighbor_features, mask) in enumerate(data_loader):
        if i > 2:
            break
        for j, sample in enumerate(batch_self_features):
            print("self: ", sample)
            print("neighbor: ", batch_neighbor_features[j])
            print("mask: ", mask[j])