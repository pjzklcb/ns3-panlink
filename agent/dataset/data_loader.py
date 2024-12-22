# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def compute_and_store_statistics(hdf5_file):
    """
    Compute mean and std of self and neighbor features, and store them in HDF5.
    """
    with h5py.File(hdf5_file, 'r+') as f:
        all_self_features = []
        all_neighbor_features = []
        for step in f.keys():
            if 'step' not in step:
                continue
            group = f[step]
            self_features = group['self_features'][:]
            neighbor_features = group['neighbor_features'][:]

            # skip the sample with no neighbor
            if len(neighbor_features) == 0:
                continue
            
            all_self_features.append(self_features)
            all_neighbor_features.extend(neighbor_features)

        total_samples = len(all_self_features)
        all_self_features = np.array(all_self_features)
        all_neighbor_features = np.array(all_neighbor_features)

        # calculate mean and std
        self_mean = np.mean(all_self_features, axis=0)
        self_std = np.std(all_self_features, axis=0)
        neighbor_mean = np.mean(all_neighbor_features, axis=0)
        neighbor_std = np.std(all_neighbor_features, axis=0)

        # save statistics
        if 'statistics' not in f:
            stats_group = f.create_group('statistics')
            stats_group.create_dataset('self_mean', data=self_mean)
            stats_group.create_dataset('self_std', data=self_std)
            stats_group.create_dataset('neighbor_mean', data=neighbor_mean)
            stats_group.create_dataset('neighbor_std', data=neighbor_std)
            stats_group.create_dataset('total_samples', data=total_samples)
        else:
            stats_group = f['statistics']
            stats_group['self_mean'][()] = self_mean
            stats_group['self_std'][()] = self_std
            stats_group['neighbor_mean'][()] = neighbor_mean
            stats_group['neighbor_std'][()] = neighbor_std
            stats_group['total_samples'][()] = total_samples

        print("Statistics computed and stored in", hdf5_file)

    return self_mean, self_std, neighbor_mean, neighbor_std, total_samples


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, selected_self_features=None, selected_neighbor_features=None):
        """
        Initalize the dataset from the HDF5 file.
        """
        self.hdf5_file = hdf5_file
        self.h5f = None # lazy loading
        with h5py.File(hdf5_file, 'r') as f:
            stats = f['statistics']
            self.self_mean = stats['self_mean'][:]
            self.self_std = stats['self_std'][:]
            self.neighbor_mean = stats['neighbor_mean'][:]
            self.neighbor_std = stats['neighbor_std'][:]
            self.total_samples = stats['total_samples'][()]
            self.steps = list(f.keys())
            self.steps.remove('statistics')

            # get selected feature indices
            self_feature_names = f['step_0/self_features'].attrs.get('feature_names')
            neighbor_feature_names = f['step_0/neighbor_features'].attrs.get('feature_names')
            self.selected_self_indices = np.arange(len(self_feature_names))
            self.selected_neighbor_indices = np.arange(len(neighbor_feature_names))
            if selected_self_features is not None:
                self.selected_self_indices = [
                    i for i, name in enumerate(self_feature_names)
                    if name in selected_self_features
                ]
            if selected_neighbor_features is not None:
                self.selected_neighbor_indices = [
                    i for i, name in enumerate(neighbor_feature_names)
                    if name in selected_neighbor_features
                ]
    
    def _init_hdf5(self):
        """
        Make sure the HDF5 file is opened independently in each worker.
        """
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Return a normalized sample from the dataset.
        """
        self._init_hdf5()
        group = self.h5f[self.steps[idx]]
        self_features = group['self_features'][:]
        neighbor_features = group['neighbor_features'][:]
        min_neighbor_backoff = group['min_neighbor_backoff'][()]

        # normalize
        self_features = (self_features - self.self_mean) / (self.self_std + 1e-8)
        neighbor_features = (neighbor_features - self.neighbor_mean) / (self.neighbor_std + 1e-8)

        # pick out selected features
        self_features = self_features[self.selected_self_indices]
        neighbor_features = neighbor_features[:, self.selected_neighbor_indices]

        return {
            'self_features': torch.tensor(self_features, dtype=torch.float32).unsqueeze(0), # (1, self_dim)
            'neighbor_features': torch.tensor(neighbor_features, dtype=torch.float32),
            'min_neighbor_backoff': torch.tensor(min_neighbor_backoff, dtype=torch.float32)
        }

    def __del__(self):
        if self.h5f is not None:
            try:
                self.h5f.close()
            except Exception as e:
                print(f"Error closing HDF5 file: {e}")


def collate_fn(batch):
    """
    Padding to have same dimension and masking invalid neighbor features
    """
    # Extract individual elements
    self_features = torch.stack([item['self_features'] for item in batch])
    min_neighbor_backoff = torch.stack([item['min_neighbor_backoff'] for item in batch])
    
    neighbor_features = [item['neighbor_features'] for item in batch]
    neighbor_lengths = [nf.shape[0] for nf in neighbor_features]

    # Pad neighbor_features to the maximum length in the batch
    padded_neighbor_features = torch.nn.utils.rnn.pad_sequence(
        neighbor_features, batch_first=True, padding_value=0.0
    )  # Shape: (batch_size, max_neighbors, neighbor_feature_dim)

    # Create key_padding_mask
    max_neighbors = padded_neighbor_features.size(1)
    key_padding_mask = torch.zeros((len(batch), max_neighbors), dtype=torch.bool)
    for i, length in enumerate(neighbor_lengths):
        key_padding_mask[i, length:] = True

    return self_features, padded_neighbor_features, key_padding_mask, min_neighbor_backoff


if __name__ == '__main__':
    # compute the statistics and store them in the dataset
    # file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1734690557.h5"
    # self_mean, self_std, neighbor_mean, neighbor_std, total_samples = compute_and_store_statistics(file_path)
    # print(f"self_mean: {self_mean}, self_std: {self_std}, "
    #       f"neighbor_mean: {neighbor_mean}, neighbor_std: {neighbor_std}, "
    #       f"total_samples: {total_samples}")

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1734690536.h5"
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
    
    dataset = HDF5Dataset(file_path, selected_self_features, selected_neighbor_features)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # test data loader
    for batch_idx, (self_features, padded_neighbors, key_padding_mask, min_neighbor_backoffs) in enumerate(data_loader):
        if batch_idx >= 10:
            break
        print(f"Batch {batch_idx + 1}:")
        print("Self Features Shape:", self_features.shape)
        print("Padded Neighbors Shape:", padded_neighbors.shape)
        print("Key Padding Mask Shape:", key_padding_mask.shape)
        print("Min Neighbor Backoffs Shape:", min_neighbor_backoffs.shape)