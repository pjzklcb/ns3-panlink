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
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = {}
        for row in reader:
            for i in range(len(headers)):
                if headers[i] not in data:
                    data[headers[i]] = []
                data[headers[i]].append(float(row[i]))
        
        for k, v in data.items():
            data[k] = np.array(v)
        return data

def calculate_mutual_info(x, y):
    h_x = entropy(np.unique(x, return_counts=True)[1])
    h_y = entropy(np.unique(y, return_counts=True)[1])
    h_xy = entropy(np.histogram2d(x, y, bins=[len(np.unique(x)), len(np.unique(y))])[0].flatten())

    mi = h_x + h_y - h_xy
    return mi

def plot_action(data):
    action = data['action']
    
    # plot action distribution
    plt.figure()
    plt.hist(action, bins=int(max(action))+1)
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.savefig('{}/action_dist.png'.format(result_dir))
    plt.close()
    
    # classify action space based on the dist
    labels = [str(v) for v in range(int(max(action))+1)]
    ref_ratio = [0] * len(labels)
    for i in range(len(labels)):
        if '-' in labels[i]:
            left, right = [int(v) for v in labels[i].split('-')]
        else:
            left, right = int(labels[i]), int(labels[i])
        ref_ratio[i] = np.sum((action >= left) & (action <= right))
    ref_ratio /= sum(ref_ratio)

    # calculate the mutual information between obs and action
    with open('{}/mutual_info.csv'.format(result_dir), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['obs', 'mi'])
        for key, value in data.items():
            mi = calculate_mutual_info(value, action)
            writer.writerow([key, mi])

    # plot the relationship between obs and action
    # obs_name_list = ['n1', 'n5', 'n10', 'n20', 'n50', 'n100', 'other']
    # obs_name_list = ['n10+n20', 'n10+n20+n30', 'n10+n20+n30+n50']
    # obs_name_list = ['n100+n50', 'n100+n50+n30', 'n100+n50+n30+n20']
    obs_name_list = ['nn']

    use_ratio = True
    x_ref = -0.1
    for obs_name in obs_name_list:
        if obs_name == 'other':
            obs = 1 - sum([data[name] for name in obs_name_list if name != 'other'])
        elif '+' in obs_name:
            names = obs_name.split('+')
            obs = sum([data[name] for name in names])
        else:
            obs = data[obs_name]
        
        obs_bins = np.linspace(min(obs), max(obs), 20)
        width = obs_bins[1] - obs_bins[0]
        hist, _ = np.histogram(obs, bins=obs_bins)

        counts = [np.zeros(len(obs_bins)-1) for _ in range(len(labels))]
        for i in range(len(obs_bins)-1):
            action_slice = action[(obs >= obs_bins[i]) & (obs < obs_bins[i+1])]
            if len(action_slice) > 0:
                for j in range(len(labels)):
                    if '-' in labels[j]:
                        left, right = [int(v) for v in labels[j].split('-')]
                    else:
                        left, right = int(labels[j]), int(labels[j])
                    counts[j][i] = np.sum((action_slice >= left) & (action_slice <= right))

        ratios = [np.zeros(len(obs_bins)-1) for _ in range(len(labels))]
        for i in range(len(labels)):
            ratios[i] = np.divide(counts[i], [v if v != 0 else 1 for v in hist])

        values = ratios if use_ratio else counts

        # create stacked histogram
        plt.figure()
        x = obs_bins[:-1] if not use_ratio else np.append(obs_bins[:-1], x_ref)
        bottom = np.zeros(len(obs_bins) - int(not use_ratio))
        for i in range(len(labels)):
            y = values[i] if not use_ratio else np.append(values[i], ref_ratio[i])
            plt.bar(x, y, bottom=bottom, width=width, alpha=0.8, label=labels[i])
            bottom += y

        if use_ratio:   
            plt.xticks(np.append(plt.xticks()[0], x_ref), np.append(plt.xticks()[1], 'ref'))
        plt.xlabel('Obs {}'.format(obs_name))
        plt.ylabel('Ratio' if use_ratio else 'Count')
        plt.legend(loc='upper right')
        plt.savefig('{}/obs_{}_{}_action.png'.format(result_dir, obs_name, 'ratio' if use_ratio else 'count'))
        plt.close()

def plot_reward(data):
    reward = data['reward']

    use_ratio = True

    obs_name = 'nn'
    obs = data[obs_name]
    obs_bins = np.linspace(min(obs), max(obs), 20)
    width = obs_bins[1] - obs_bins[0]
    hist, _ = np.histogram(obs, bins=obs_bins)
    counts = [np.zeros(len(obs_bins)-1) for _ in range(3)]
    for i in range(len(obs_bins)-1):
        reward_slice = reward[(obs >= obs_bins[i]) & (obs < obs_bins[i+1])]
        counts[0][i] = np.sum(reward_slice < 0)
        counts[1][i] = np.sum(reward_slice == 0)
        counts[2][i] = np.sum(reward_slice > 0)

    ratios = [np.zeros(len(obs_bins)-1) for _ in range(3)]
    for i in range(3):
        ratios[i] = np.divide(counts[i], [v if v != 0 else 1 for v in hist])

    values = ratios if use_ratio else counts

    # create stacked histogram
    plt.figure()
    x = obs_bins[:-1]
    bottom = np.zeros(len(obs_bins) - 1)
    for i in range(3):
        y = values[i]
        plt.bar(x, y, bottom=bottom, width=width, alpha=0.8, label='<0' if i == 0 else '=0' if i == 1 else '>0')
        bottom += y

    plt.xlabel('Obs {}'.format(obs_name))
    plt.ylabel('Ratio' if use_ratio else 'Count')
    plt.legend(loc='upper right')
    plt.savefig('{}/obs_{}_{}_action.png'.format(result_dir, obs_name, 'ratio' if use_ratio else 'count'))
    plt.close()


if __name__ == "__main__":
    prefix = "contrib/panlink/result-"
    timestamp = "1733798820"
    suffix = "-ideal-backoff"
    episode = 0
    result_dir = prefix + timestamp + suffix + "/episode_" + str(episode) + "/learn/"
    filename = result_dir + "debug.csv"
    data = csv_to_dict(filename)

    # plot_action(data)
    # plot_reward(data)
    backoff = data['min_neighbor_backoff']
    backoff[backoff > 15] = 16
    x = backoff
    # x = data['nn']
    for key, value in data.items():
        metric = calculate_mutual_info(x, value)
        print(f"{key:>20}  {metric:.4f}")