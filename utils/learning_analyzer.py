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


if __name__ == "__main__":
    prefix = "contrib/panlink/"
    episode = 199
    result_dir = prefix + "result-1733188616-8fix3-rm-centering/episode_" + str(episode) + "/learn/"
    filename = result_dir + "debug.csv"
    data = csv_to_dict(filename)

    # plot action distribution
    action = data['action']
    # plt.figure()
    # plt.hist(action, bins=int(max(action))+1)
    # plt.xlabel('Action')
    # plt.ylabel('Count')
    # plt.savefig('{}/action_dist.png'.format(result_dir))
    # plt.close()
    
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

    # analyze the relationship between obs and action
    use_ratio = True
    use_log_scale = False
    x_ref = 1/4 if use_log_scale else -0.1
    obs_name_list = ['n10', 'n20', 'n30', 'n50', 'n100', 'other']
    for obs_name in obs_name_list:
        if obs_name == 'other':
            obs = 1 - sum([data[name] for name in obs_name_list if name != 'other'])
        else:
            obs = data[obs_name]
        
        if use_log_scale:
            obs = [v if v != 0 else 0.5 for v in obs]
            obs_bins = np.logspace(np.log2(min(obs)), np.log2(max(obs)), 20, base=2)
            width = np.diff(obs_bins) * 0.8
        else:
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
        width = width if (not use_log_scale or not use_ratio) else np.append(width, 1/16)
        bottom = np.zeros(len(obs_bins) - int(not use_ratio))
        for i in range(len(labels)):
            y = values[i] if not use_ratio else np.append(values[i], ref_ratio[i])
            plt.bar(x, y, bottom=bottom, width=width, alpha=0.8, label=labels[i])
            bottom += y

        if use_log_scale:
            plt.xscale('log', base=2)
        if use_ratio:   
            plt.xticks(np.append(plt.xticks()[0], x_ref), np.append(plt.xticks()[1], 'ref'))
        plt.xlabel('Obs {}'.format(obs_name))
        plt.ylabel('Ratio' if use_ratio else 'Count')
        plt.legend(loc='upper right')
        plt.savefig('{}/obs_{}_{}_action.png'.format(result_dir, obs_name, 'ratio' if use_ratio else 'count'))
        plt.close()