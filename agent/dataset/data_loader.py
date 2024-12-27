# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import time
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def dataset_transform(hdf5_file, epsilon=1e-8):
    """
    Change the format of the dataset and store them in a new HDF5.
    """
    output_file = hdf5_file.replace('.h5', '_trans.h5')

    with h5py.File(hdf5_file, 'r') as f:
        self_feature_name = f['step_0/self_features'].attrs.get('feature_names')
        neighbor_feature_name = f['step_0/neighbor_features'].attrs.get('feature_names')

        num_steps = len([key for key in f.keys() if 'step' in key])
        max_neighbors_per_step = 20 
        all_self_features = np.zeros((num_steps, len(self_feature_name)))
        all_neighbor_features = np.zeros((num_steps * max_neighbors_per_step, len(neighbor_feature_name)))

        neighbor_feature_idx = np.zeros((num_steps, 2), dtype=int)
        min_neighbor_backoff = np.zeros(num_steps)
        cnt = 0
        valid_num_steps = 0
        for step in f.keys():
            if 'step' not in step:
                continue
            group = f[step]
            self_features = group['self_features'][:]
            neighbor_features = group['neighbor_features'][:]

            # skip the sample with no neighbor
            if len(neighbor_features) == 0:
                continue
            
            all_self_features[valid_num_steps] = self_features
            all_neighbor_features[cnt:cnt+len(neighbor_features)] = neighbor_features
            neighbor_feature_idx[valid_num_steps] = [cnt, cnt+len(neighbor_features)]
            min_neighbor_backoff[valid_num_steps] = group['min_neighbor_backoff'][()]

            cnt += len(neighbor_features)
            valid_num_steps += 1

        all_neighbor_features = all_neighbor_features[:cnt]

        s_df = pd.DataFrame(all_self_features, columns=self_feature_name)
        n_df = pd.DataFrame(all_neighbor_features, columns=neighbor_feature_name)

        # preprocessing
        tx_range = s_df['tx_interval_max'] - s_df['tx_interval_min']
        self_data = {
            # self only features
            'delta_time': s_df['delta_time'],
            'cw': np.log2(1 + s_df['cw']) / 10,
            'n_defer_access': s_df['n_defer_access'],
            'n_waited_slots': s_df['n_waited_slots'],
            'n_in_queue': s_df['n_in_queue'],
            'n_sending': s_df['n_sending'],
            'n_acked': s_df['n_acked'] / (s_df['n_sending'] + epsilon),

            'n_tx_data': s_df['n_tx_data'],
            'n_acked_data': s_df['n_acked_data'] / (s_df['n_tx_data'] + epsilon),
            'last_txok_delay': s_df['last_txok_delay'],
            'last_tx_delay': s_df['last_tx_delay'],
            'chan_util': s_df['chan_util'],
            'n_rx_fail': s_df['n_rx_fail'],
            'idle_ratio': s_df['idle_ratio'],
            'tx_interval_mean': s_df['tx_interval_mean'],
            'tx_interval_std': s_df['tx_interval_std'],
            'tx_prob': np.where(
                s_df['tx_interval_mean'] > 0, np.log2(1 + s_df['last_txok_delay']/s_df['tx_interval_mean']), 0
            ),
            'tx_range_to_mean': tx_range / (s_df['tx_interval_mean'] + epsilon),
            'tx_cv': s_df['tx_interval_std'] / (s_df['tx_interval_mean'] + epsilon),
            'tx_center_position': (s_df['tx_interval_mean'] - s_df['tx_interval_min']) / (tx_range + epsilon),
        }

        tx_range = n_df['tx_interval_max'] - n_df['tx_interval_min']
        neighbor_data = {
            'n_tx_data': n_df['n_tx_data'],
            'n_acked_data': n_df['n_acked_data'] / (n_df['n_tx_data'] + epsilon),
            'last_txok_delay': n_df['last_txok_delay'],
            'last_tx_delay': n_df['last_tx_delay'],
            'chan_util': n_df['chan_util'],
            'tx_interval_mean': n_df['tx_interval_mean'],
            'tx_interval_std': n_df['tx_interval_std'],
            'tx_prob': np.where(
                n_df['tx_interval_mean'] > 0, np.log2(1 + n_df['last_txok_delay']/n_df['tx_interval_mean']), -1
            ),
            'tx_range_to_mean': tx_range / (n_df['tx_interval_mean'] + epsilon),
            'tx_cv': n_df['tx_interval_std'] / (n_df['tx_interval_mean'] + epsilon),
            'tx_center_position': (n_df['tx_interval_mean'] - n_df['tx_interval_min']) / (tx_range + epsilon),
            'backoff': n_df['backoff'],
        }

        # normalization
        s_df_extend = pd.DataFrame(self_data)
        n_df_extend = pd.DataFrame(neighbor_data)

        # self_only_name = ['delta_time', 'n_in_queue', 'n_sending', 'n_rx_drop']
        # common_name = ['n_tx_data', 'last_txok_delay', 'chan_util', 'tx_interval_mean', 'tx_interval_std']
        # s_df_extend[self_only_name] = s_df_extend[self_only_name] / (s_df_extend[self_only_name].max() + epsilon)
        # column_max = np.maximum(s_df_extend[common_name].max(), n_df_extend[common_name].max())
        # s_df_extend[common_name] = s_df_extend[common_name] / (column_max + epsilon)
        # n_df_extend[common_name] = n_df_extend[common_name] / (column_max + epsilon)
    
    # save to new hdf5 file
    with h5py.File(output_file, 'w') as o:
        # create dataset
        o.create_dataset('self_features', data=s_df_extend.to_numpy()) # (total_samples, selected_self_dim)
        o.create_dataset('neighbor_features', data=n_df_extend.to_numpy()) # (total_samples, N*selected_neighbor_dim)
        o.create_dataset('neighbor_feature_idx', data=neighbor_feature_idx) # (total_samples, 2)
        o.create_dataset('min_neighbor_backoff', data=min_neighbor_backoff) # (total_samples,)
        o['self_features'].attrs['feature_names'] = s_df_extend.columns.tolist()
        o['neighbor_features'].attrs['feature_names'] = n_df_extend.columns.tolist()

    print('Dataset transformation done.')


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, selected_self_features=None, selected_neighbor_features=None):
        """
        Initalize the dataset from the HDF5 file.
        """
        self.hdf5_file = hdf5_file
        self.h5f = None # lazy loading
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['min_neighbor_backoff'][:])
            # get feature names
            self_feature_names = f['self_features'].attrs.get('feature_names')
            neighbor_feature_names = f['neighbor_features'].attrs.get('feature_names')

            if selected_self_features is None:
                self.selected_self_indices = list(range(len(self_feature_names)))
            else:
                self.selected_self_indices = [
                    i for i, name in enumerate(self_feature_names) if name in selected_self_features
                ]
            if selected_neighbor_features is None:
                self.selected_neighbor_indices = list(range(len(neighbor_feature_names)))
            else:
                self.selected_neighbor_indices = [
                    i for i, name in enumerate(neighbor_feature_names) if name in selected_neighbor_features
                ]
    
    def _init_hdf5(self):
        """
        Make sure the HDF5 file is opened independently in each worker.
        """
        if self.h5f is None:
            self.h5f = h5py.File(self.hdf5_file, 'r')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Return a normalized sample from the dataset.
        """
        self._init_hdf5()

        self_features = self.h5f['self_features'][idx]
        neighbor_feature_idx = self.h5f['neighbor_feature_idx'][idx]
        neighbor_features = self.h5f['neighbor_features'][neighbor_feature_idx[0]:neighbor_feature_idx[1]]
        min_neighbor_backoff = self.h5f['min_neighbor_backoff'][idx]

        selected_self_features = self_features[self.selected_self_indices]
        selected_neighbor_features = neighbor_features[:, self.selected_neighbor_indices]
        return {
            'self_features': torch.tensor(selected_self_features, dtype=torch.float32).unsqueeze(0), # (1, self_dim)
            'neighbor_features': torch.tensor(selected_neighbor_features, dtype=torch.float32),
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
    # preprocess features and store them in a new dataset
    # tic = time.time()
    # file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735182828.h5"
    # dataset_transform(file_path)
    # toc = time.time() - tic
    # print(f"Cost {toc // 60}min {toc % 60}s")

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735182819_trans.h5"
    selected_self_features = [#'delta_time', 'cw', 'n_defer_access', 'n_waited_slots', 
                            #   'n_in_queue', 'n_sending', 'n_acked', 'n_rx_drop', 
                            #   'n_tx_data', 'n_acked_data', #'last_tx_delay', 
                            #   'last_txok_delay', 'chan_util', 
                            #   'app_interval_mean', 'app_interval_std', 'app_interval_min', 'app_interval_max',
                            #   'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                            #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                              'tx_prob', #'tx_range_to_mean', 'tx_cv', 'tx_center_position'
                              ]
    selected_neighbor_features = [#'role', 
                                  'n_tx_data', 'n_acked_data', #'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', #'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                                #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                                  'tx_prob', 'tx_range_to_mean', 'tx_cv', 'tx_center_position'
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
    
    # # analyze dataset
    # feature_list = []
    labels = np.zeros(len(data_loader.dataset))
    cnt = 0
    for feature, _, _, min_neighbor_backoff in data_loader:
        batch_size = feature.shape[0]
        # batch_feature = feature.view(-1)
        # batch_feature = feature.view(-1, feature.shape[-1])
        # batch_feature = feature.squeeze(1)
        # feature_list.extend(batch_feature.tolist())
        labels[cnt:cnt + batch_size] = min_neighbor_backoff.numpy()
        cnt += batch_size
    
    print(sum(labels>0)/len(labels))
    # y = feature_list

    # plt.figure()
    # plt.hist(y, bins=20)
    # plt.show()