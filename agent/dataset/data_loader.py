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
            'delta_time': np.log(1 + s_df['delta_time']),
            'cw': np.log2(1 + s_df['cw']) / 10,
            'n_defer_access': np.minimum(s_df['n_defer_access'] / (s_df['cw'] + 1), 1),
            'n_waited_slots': np.minimum(s_df['n_waited_slots'] / (s_df['cw'] + 1), 1),
            'n_in_queue': s_df['n_in_queue'],
            'n_sending': s_df['n_sending'],
            'n_acked': s_df['n_acked'] / (s_df['n_sending'] + epsilon),
            'n_rx_drop': np.log(1 + s_df['n_rx_drop']/(s_df['delta_time'] + epsilon)),

            'n_tx_data': s_df['n_tx_data'],
            'n_acked_data': s_df['n_acked_data'] / (s_df['n_tx_data'] + epsilon),
            'last_txok_delay': np.log(1 + s_df['last_txok_delay']),
            'chan_util': s_df['chan_util'],
            'tx_interval_mean': np.log(1 + s_df['tx_interval_mean']),
            'tx_interval_std': np.log(1 + s_df['tx_interval_std']),
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
            'last_txok_delay': np.log(1 + n_df['last_txok_delay']),
            'chan_util': n_df['chan_util'],
            'tx_interval_mean': np.log(1 + n_df['tx_interval_mean']),
            'tx_interval_std': np.log(1 + n_df['tx_interval_std']),
            'tx_prob': np.where(
                n_df['tx_interval_mean'] > 0, np.log2(1 + n_df['last_txok_delay']/n_df['tx_interval_mean']), -1
            ),
            'tx_range_to_mean': tx_range / (n_df['tx_interval_mean'] + epsilon),
            'tx_cv': n_df['tx_interval_std'] / (n_df['tx_interval_mean'] + epsilon),
            'tx_center_position': (n_df['tx_interval_mean'] - n_df['tx_interval_min']) / (tx_range + epsilon),
        }

        # normalization
        s_df_extend = pd.DataFrame(self_data)
        n_df_extend = pd.DataFrame(neighbor_data)

        self_only_name = ['delta_time', 'n_in_queue', 'n_sending', 'n_rx_drop']
        common_name = ['n_tx_data', 'last_txok_delay', 'chan_util', 'tx_interval_mean', 'tx_interval_std']
        s_df_extend[self_only_name] = s_df_extend[self_only_name] / (s_df_extend[self_only_name].max() + epsilon)
        column_max = np.maximum(s_df_extend[common_name].max(), n_df_extend[common_name].max())
        s_df_extend[common_name] = s_df_extend[common_name] / (column_max + epsilon)
        n_df_extend[common_name] = n_df_extend[common_name] / (column_max + epsilon)
    
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


def preprocess_step(hdf5_file, step, self_feature_name, neighbor_feature_name, 
                    self_time_names, time_feature_names, stats, epsilon):
    """
    Preprocess a single step and return the normalized data.
    """
    with h5py.File(hdf5_file, 'r') as f:
        group = f[step]
        self_features = group['self_features'][:]
        neighbor_features = group['neighbor_features'][:]
        min_neighbor_backoff = group['min_neighbor_backoff'][()]

        # Skip samples with no neighbors
        if len(neighbor_features) == 0:
            return None

        self_df = pd.DataFrame([self_features], columns=self_feature_name)
        neighbor_df = pd.DataFrame(neighbor_features, columns=neighbor_feature_name)

        # Apply normalization 
        self_df_norm = self_df.copy()  # Normalize self_features
        neighbor_df_norm = neighbor_df.copy()  # Normalize neighbor_features

        self_df_norm['cw'] = np.log2(1 + self_df['cw']) / 10
        self_df_norm['n_defer_access'] = np.minimum(self_df['n_defer_access']/(self_df['cw']+1), 1)
        self_df_norm['n_waited_slots'] = np.minimum(self_df['n_waited_slots']/(self_df['cw']+1), 1)
        self_df_norm['n_in_queue'] = self_df['n_in_queue'] / (stats['max_n_in_queue'] + epsilon)
        self_df_norm['n_sending'] = self_df['n_sending'] / (stats['max_n_sending'] + epsilon)
        self_df_norm['n_acked'] = self_df['n_acked'] / (self_df['n_sending'] + epsilon)
        self_df_norm['n_tx_data'] = (
            (self_df['n_tx_data'] - stats['mean_self_n_tx_data']) / (stats['std_self_n_tx_data'] + epsilon)
        )
        self_df_norm['n_acked_data'] = self_df['n_acked_data'] / (self_df['n_tx_data'] + epsilon)
        self_df_norm['chan_util'] = self_df['chan_util'] / 500
        self_df_norm[self_time_names + time_feature_names] = (
            (np.log(1 + self_df[self_time_names + time_feature_names]) - stats['mean_self_time_log']) / (
                stats['std_self_time_log'] + epsilon)
        )

        neighbor_df_norm['role'] = neighbor_df['role'] / 3
        neighbor_df_norm['n_tx_data'] = (
            (neighbor_df['n_tx_data'] - stats['mean_neighbor_n_tx_data']) / (
                stats['std_neighbor_n_tx_data'] + epsilon)
        )
        neighbor_df_norm['n_acked_data'] = neighbor_df['n_acked_data'] / (neighbor_df['n_tx_data'] + epsilon)
        neighbor_df_norm['chan_util'] = neighbor_df['chan_util'] / 500
        neighbor_df_norm['mean_rssi'] = (
            (neighbor_df['mean_rssi'].replace(0, -40) - stats['mean_rssi']) / (stats['std_rssi'] + epsilon)
        )
        neighbor_df_norm[time_feature_names] = (
            (np.log(1 + neighbor_df[time_feature_names]) - stats['mean_neighbor_time_log']) / (
                stats['std_neighbor_time_log'] + epsilon)
        )

    return step, self_df_norm, neighbor_df_norm, min_neighbor_backoff


def preprocess_and_store_statistics(hdf5_file, epsilon=1e-8):
    """
    Preprocess the dataset and store them in a new HDF5.
    """
    output_file = hdf5_file.replace('.h5', '_norm.h5')

    with h5py.File(hdf5_file, 'r') as f:
        self_feature_name = f['step_0/self_features'].attrs.get('feature_names')
        neighbor_feature_name = f['step_0/neighbor_features'].attrs.get('feature_names')

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

        self_df = pd.DataFrame(all_self_features, columns=self_feature_name)
        neighbor_df = pd.DataFrame(all_neighbor_features, columns=neighbor_feature_name) 

        # calculate normalization stats
        self_time_names = ['delta_time', 'n_rx_drop', 
                           'app_interval_mean', 'app_interval_std', 'app_interval_min', 'app_interval_max']
        time_feature_names = ['last_tx_delay', 'last_txok_delay', 
                            'tx_interval_mean', 'tx_interval_std', 'tx_interval_min', 'tx_interval_max', 
                            'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max']
        self_time_log = np.log(1 + self_df[self_time_names + time_feature_names])
        neighbor_time_log = np.log(1 + neighbor_df[time_feature_names])

        stats = {
            'max_n_in_queue': self_df['n_in_queue'].max(),
            'max_n_sending': self_df['n_sending'].max(),
            'mean_self_n_tx_data': self_df['n_tx_data'].mean(),
            'std_self_n_tx_data': self_df['n_tx_data'].std(),
            'mean_self_time_log': self_time_log.mean(),
            'std_self_time_log': self_time_log.std(),

            'mean_neighbor_n_tx_data': neighbor_df['n_tx_data'].mean(),
            'std_neighbor_n_tx_data': neighbor_df['n_tx_data'].std(),
            'mean_rssi': neighbor_df['mean_rssi'].replace(0, -40).mean(),
            'std_rssi': neighbor_df['mean_rssi'].replace(0, -40).std(),
            'mean_neighbor_time_log': neighbor_time_log.mean(),
            'std_neighbor_time_log': neighbor_time_log.std(),
        }

        # Process each step in parallel
        steps = [step for step in f.keys() if 'step' in step]
        results = []

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(preprocess_step, hdf5_file, step, self_feature_name, neighbor_feature_name, 
                                self_time_names, time_feature_names, stats, epsilon)
                for step in steps
            ]

            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)
        
    with h5py.File(output_file, 'w') as o:
        for step, self_df_norm, neighbor_df_norm, min_neighbor_backoff in results:
            # create dataset
            o_group = o.create_group(step)
            o_group.create_dataset('self_features', data=self_df_norm.to_numpy()[0]) # (self_dim,)
            o_group.create_dataset('neighbor_features', data=neighbor_df_norm.to_numpy())
            o_group.create_dataset('min_neighbor_backoff', data=min_neighbor_backoff)
            o_group['self_features'].attrs['feature_names'] = self_df_norm.columns.to_list()
            o_group['neighbor_features'].attrs['feature_names'] = neighbor_df_norm.columns.to_list()

    print('Dataset preprocessing done.')


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
    # file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735005131.h5"
    # dataset_transform(file_path)
    # toc = time.time() - tic
    # print(f"Cost {toc // 60}min {toc % 60}s")

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735005131_trans.h5"
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
    feature_list = []
    for feature, _, _, _ in data_loader:
        batch_feature = feature.view(-1)
        # batch_feature = feature.view(-1, feature.shape[-1])
        # batch_feature = feature.squeeze(1)
        feature_list.extend(batch_feature.tolist())
    
    y = feature_list
    # # y = np.log(np.array(feature_list) + 1)
    # # log_feature = np.log(np.array(feature_list) + 1)
    # # mean = np.mean(log_feature)
    # # std = np.std(log_feature)
    # # y = (log_feature - mean) / std

    # # last_tx_delay, last_txok_delay = zip(*feature_list)
    # # a = np.array(last_tx_delay)
    # # b = np.array(last_txok_delay)
    # # y = np.where(b > 0, a/b, -1)

    plt.figure()
    plt.hist(y, bins=20)
    plt.show()