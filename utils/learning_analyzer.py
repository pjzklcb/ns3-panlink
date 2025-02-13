# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import entropy
from sklearn.metrics import r2_score, normalized_mutual_info_score


def csv_to_dict(filename):
    df = pd.read_csv(filename)
    groups = []
    current_group = None
    for _, row in df.iterrows():
        if pd.isna(row["sim_time"]):
            neighbor = {
                "node_id": int(row["node_id"]),
                "backoff": row["backoff"],
                "txok_delay": row["txok_delay"],
                "txok_idle_time": row["txok_idle_time"]
            }
            current_group["neighbors"].append(neighbor)
        else:
            if current_group is not None:
                groups.append(current_group)

            current_group = {
                "sim_time": row["sim_time"],
                "n_sending": row["n_sending"],
                "n_acked": row["n_acked"],
                "cw": row["cw"],
                "n_deferred": row["n_deferred"],
                "n_waited_slots": row["n_waited_slots"],
                "n_contention": row["n_contention"],
                "txok_delay": row["backoff"],
                "chan_fail_delay": row["txok_delay"],
                "chan_fail_idle_time": row["txok_idle_time"],
                "pk": row["pk"],
                "action": row["action"],
                "new_backoff": row["new_backoff"],
                "reward": row["reward"],
                "neighbors": []
            }

    if current_group is not None:
        groups.append(current_group)
    return groups


if __name__ == "__main__":
    prefix = "contrib/panlink/result-"
    timestamp = "1739361635"
    suffix = "-saturated4"
    episode = 200
    result_dir = prefix + timestamp + suffix + "/episode_" + str(episode) + "/learn/"
    filename = result_dir + "debug.csv"
    groups = csv_to_dict(filename)

    max_neighbor_waited_slots = []
    min_neighbor_backoff = []
    pk = []
    for group in groups:
        pk.append(group["pk"])
        current_idle_time = 0
        current_backoff = 1024
        for n in group["neighbors"]:
            current_backoff = min(current_backoff, n["backoff"])
            if n["node_id"] != 0:
                current_idle_time = max(current_idle_time, n["txok_idle_time"])
        max_neighbor_waited_slots.append(min(current_idle_time, group["chan_fail_idle_time"]) / 0.09)
        min_neighbor_backoff.append(current_backoff)
    
    plt.figure()
    # plt.hist([v for v in min_neighbor_backoff if v < 20], bins=20)
    # plt.hist([v for v in max_neighbor_waited_slots if v < 40], bins=40)

    # plt.hist([pk[i] for i in range(len(min_neighbor_backoff)) if min_neighbor_backoff[i] <= 2], 
    #          bins=40, density=True, alpha=0.5, label='min_neighbor_backoff <= 2')
    # plt.hist([pk[i] for i in range(len(min_neighbor_backoff)) if min_neighbor_backoff[i] > 2], 
    #          bins=40, density=True, alpha=0.5, label='min_neighbor_backoff > 2')
    plt.hist([pk[i] for i in range(len(max_neighbor_waited_slots)) if max_neighbor_waited_slots[i] <= 3], 
             bins=40, density=True, alpha=0.5, label='max_neighbor_waited_slots <= 3')
    plt.hist([pk[i] for i in range(len(max_neighbor_waited_slots)) if max_neighbor_waited_slots[i] > 3], 
             bins=40, density=True, alpha=0.5, label='max_neighbor_waited_slots > 3')
    plt.legend()
    plt.xlabel("Probability of choosing action=0")
    plt.ylabel("Density")

    plt.show()