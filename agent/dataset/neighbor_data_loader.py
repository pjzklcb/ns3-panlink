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


def dataset_transform(hdf5_file, epsilon=1e-8):
    """
    Change the format of the dataset and store them in a new HDF5.
    """
    output_file = hdf5_file.replace('.h5', '_neighbor.h5')

    with h5py.File(hdf5_file, 'r') as f:
        neighbor_feature_name = f['step_0/neighbor_features'].attrs.get('feature_names')

        num_steps = len([key for key in f.keys() if 'step' in key])
        max_neighbors_per_step = 20 # the maximum number of neighbors per step
        all_neighbor_features = np.zeros((num_steps * max_neighbors_per_step, len(neighbor_feature_name)))

        cnt = 0
        for step in f.keys():
            if 'step' not in step:
                continue
            group = f[step]
            neighbor_features = group['neighbor_features'][:]

            # skip the sample with no neighbor
            if len(neighbor_features) == 0:
                continue
            
            all_neighbor_features[cnt:cnt+len(neighbor_features)] = neighbor_features

            cnt += len(neighbor_features)

        all_neighbor_features = all_neighbor_features[:cnt]

        n_df = pd.DataFrame(all_neighbor_features, columns=neighbor_feature_name)

        # preprocessing
        tx_range = n_df['tx_interval_max'] - n_df['tx_interval_min']
        extend_feature = {
            'r_acked_data': n_df['n_acked_data'] / (n_df['n_tx_data'] + epsilon),
            'tx_prob': np.where(
                n_df['tx_interval_mean'] > 0, np.log2(1 + n_df['last_txok_delay']/n_df['tx_interval_mean']), -1
            ),
            'tx_range_to_mean': tx_range / (n_df['tx_interval_mean'] + epsilon),
            'tx_cv': n_df['tx_interval_std'] / (n_df['tx_interval_mean'] + epsilon),
            'tx_center_position': (n_df['tx_interval_mean'] - n_df['tx_interval_min']) / (tx_range + epsilon),
        }

        # normalization
        n_df_extend = pd.concat([n_df.copy(), pd.DataFrame(extend_feature)], axis=1).drop(columns=['backoff'])
        backoff = n_df['backoff']
    
    # save to new hdf5 file
    with h5py.File(output_file, 'w') as o:
        # create dataset
        o.create_dataset('neighbor_features', data=n_df_extend.to_numpy()) # (total_samples, N*selected_neighbor_dim)
        o.create_dataset('backoff', data=backoff.to_numpy()) # (total_samples,)
        o['neighbor_features'].attrs['feature_names'] = n_df_extend.columns.tolist()

    print('Dataset transformation done.')


class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file, selected_neighbor_features=None):
        """
        Initalize the dataset from the HDF5 file.
        """
        self.hdf5_file = hdf5_file
        self.h5f = None # lazy loading
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['neighbor_features'][:])
            # get feature names
            neighbor_feature_names = f['neighbor_features'].attrs.get('feature_names')

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

        neighbor_features = self.h5f['neighbor_features'][idx]
        backoff = self.h5f['backoff'][idx]

        selected_neighbor_features = neighbor_features[self.selected_neighbor_indices]
        return (
            torch.tensor(selected_neighbor_features, dtype=torch.float32), # (selected_neighbor_dim, )
            torch.tensor(backoff, dtype=torch.float32)
        )

    def __del__(self):
        if self.h5f is not None:
            try:
                self.h5f.close()
            except Exception as e:
                print(f"Error closing HDF5 file: {e}")


if __name__ == '__main__':
    # preprocess features and store them in a new dataset
    # tic = time.time()
    # file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735121289.h5"
    # dataset_transform(file_path)
    # toc = time.time() - tic
    # print(f"Cost {toc // 60}min {toc % 60}s")

    file_path = os.path.dirname(os.path.abspath(__file__)) + "/data-1735121289_neighbor.h5"
    selected_neighbor_features = [#'role', 
                                  'n_tx_data', #'n_acked_data', 'last_tx_delay', 
                                  'last_txok_delay', 'chan_util', #'mean_rssi', 
                                  'tx_interval_mean', 'tx_interval_std', #'tx_interval_min', 'tx_interval_max', 
                                #   'tx_duration_mean', 'tx_duration_std', 'tx_duration_min', 'tx_duration_max',
                                  'r_acked_data', 'tx_prob', 'tx_range_to_mean', 'tx_cv', 'tx_center_position'
                                  ]
    
    dataset = HDF5Dataset(file_path, selected_neighbor_features)
    data_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    # test data loader
    for batch_idx, (n_neighbors, backoff) in enumerate(data_loader):
        if batch_idx >= 1:
            break
        print(f"Batch {batch_idx + 1}:")
        print("Neighbors Shape:", n_neighbors.shape)
        print("Backoffs Shape:", backoff.shape)
    
    # analyze dataset
    # feature_list = np.zeros((len(data_loader.dataset), len(selected_neighbor_features)))
    # labels = []
    # cnt = 0
    # for feature, label in data_loader:
    #     batch_size = feature.shape[0]
    #     feature_list[cnt:cnt + batch_size] = feature.numpy()
    #     labels.extend(label.tolist())
    
    # labels = np.array(labels)
    # y = feature_list.reshape(-1)

    # plt.figure()
    # plt.hist(y, bins=20)
    # plt.show()