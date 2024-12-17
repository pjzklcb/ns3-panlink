# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import h5py
import pandas as pd

# 读取数据
dataset_dir = os.path.dirname(os.path.abspath(__file__))
with h5py.File(dataset_dir + "/data-1734417926.h5", "r") as f:
    print(len(f))
    self_features = f["step_0/self_features"][:]
    self_feature_names = f["step_0/self_features"].attrs.get("feature_names")
    print("Self Feature Names:", self_feature_names)
    print("Self Feature Values:", self_features)

    neighbor_features = f["step_0/neighbor_features"][:]
    neighbor_feature_names = f["step_0/neighbor_features"].attrs.get("feature_names")
    print("Neighbor Feature Names:", neighbor_feature_names)
    print("Neighbor Feature Values:", neighbor_features)

df_self = pd.DataFrame([self_features], columns=self_feature_names)
print("Self Feature DataFrame:")
print(df_self)

# 将 neighbor feature 数据转换为 DataFrame
df_neighbors = pd.DataFrame(neighbor_features, columns=neighbor_feature_names)
print("Neighbor Feature DataFrame:")
print(df_neighbors)