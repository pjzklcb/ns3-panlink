# Copyright (c) 2025
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as line2D
import os
import csv
import json
from plot_results import InFlightPacketInfo

path = os.path.abspath(os.path.dirname(__file__))

def plot_figure4(result_dir_list):
    n_episodes = 200
    r_tx_list = []
    backoff_ratio_list = []
    lambda_list = []
    for result_dir in result_dir_list:
        with open(os.path.join(result_dir, "episode.csv"), "r") as f:
            reader = csv.reader(f)
            next(reader)
            next(reader)
            r_tx = [float(x) for x in next(reader)]
            backoff_ratio = [float(x) for x in next(reader)]
            lambda_ = [float(x) for x in next(reader)]

        r_tx_list.append(r_tx[:n_episodes])
        backoff_ratio_list.append(backoff_ratio[:n_episodes])
        lambda_list.append(lambda_[:n_episodes])
    
    plt.figure(figsize=(6, 4.8))
    plt.plot(np.mean(r_tx_list, axis=0), 'b-', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Reward per TX')
    plt.tight_layout()
    plt.savefig(path + "/fig4a.png")
    plt.close()

    plt.figure(figsize=(6, 4.8))
    plt.plot(np.mean(backoff_ratio_list, axis=0), 'g-', linewidth=1)
    plt.hlines(0.5, xmin=0, xmax=n_episodes, colors='k', linestyles='dashed')
    plt.xlabel('Episode')
    plt.ylabel('Backoff Ratio')
    plt.tight_layout()
    plt.savefig(path + "/fig4b.png")
    plt.close()

    plt.figure(figsize=(6, 4.8))
    plt.plot(np.mean(lambda_list, axis=0), 'r-', linewidth=1)
    plt.xlabel('Episode')
    plt.ylabel('Lambda')
    plt.tight_layout()
    plt.savefig(path + "/fig4c.png")
    plt.close()


def plot_figure5(result_dir1, result_dir2):

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # plot left
    plot_subfigure(result_dir1, ax1)
    ax1.hlines(10, xmin=0, xmax=6, colors='k', linestyles='dashed')
    ax1.hlines(41, xmin=0, xmax=6, colors='k', linestyles='dashed')
    ax1.set_ylim([0, 50])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Throughput (Mbps)")  # 只在左侧显示Y轴标签
    ax1.legend()

    # plot right
    plot_subfigure(result_dir2, ax2)
    ax2.hlines(10, xmin=0, xmax=6, colors='k', linestyles='dashed')
    ax2.hlines(11, xmin=0, xmax=6, colors='k', linestyles='dashed')
    ax2.hlines(42, xmin=0, xmax=6, colors='k', linestyles='dashed')
    ax2.set_ylim([0, 50])
    ax2.set_xlabel("Time (s)")
    ax2.legend()

    # remove ticks from the right side
    ax2.tick_params(labelleft=False)          # 隐藏刻度标签
    # ax2.yaxis.set_ticks_position('none')      # 隐藏刻度线（可选）
    plt.subplots_adjust(wspace=0.05)          # 缩小子图间距至5%默认值
    plt.tight_layout()
    plt.savefig(path + "/fig5.png")
    plt.close(fig)

def plot_subfigure(result_dir, ax):
    csv_files = [f for f in os.listdir(result_dir+'csv/') if f.endswith('.csv')]

    # read node_*_packet.csv based on bss map
    bss_map = get_bss_map(result_dir)
    for ap_id, sta_list in bss_map.items():
        bss_packet_map = {}

        csv_file = "node_" + str(ap_id) + "_packet.csv"
        assert csv_file in csv_files, "CSV file {} not found".format(csv_file)
        bss_packet_map[ap_id] = read_packet_map_from_csv(result_dir, csv_file, ap_id)
        
        for sta_id in sta_list:
            csv_file = "node_" + str(sta_id) + "_packet.csv"
            assert csv_file in csv_files, "CSV file {} not found".format(csv_file)
            bss_packet_map[sta_id] = read_packet_map_from_csv(result_dir, csv_file, sta_id)

        # plot l2 thpt
        plot_bss_l2_thpt(bss_packet_map, ax)

def get_bss_map(result_dir):
    bss_map = {}
    with open(result_dir+'/topology.json', 'r') as f:
        doc = json.load(f)
        nodes_info = doc["nodes"]

        for node_info in nodes_info:
            bss_id = int(node_info["bssid"])
            node_id = int(node_info["id"])
            node_type = node_info["type"]
            if node_type == "PanlinkAp":
                bss_map[bss_id] = []
            elif node_type == "PanlinkSta":
                bss_map[bss_id].append(node_id)
            elif node_type == "Ap":
                bss_map[bss_id] = []
            elif node_type == "Sta":
                bss_map[bss_id].append(node_id)

    return bss_map

def read_packet_map_from_csv(result_dir, csv_file, node_id):
    print("Processing packet map for node_{}".format(node_id))
    packet_map = {}   # packet map for node_id {uid: InFlighPacketInfo, ...}
    with open(result_dir+'csv/'+csv_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            # parse packet info
            uid = int(row[0])
            phy_tx_time = [float(x)*1e-3 for x in row[13:15]]   # in seconds
            if uid not in packet_map:
                size, port, burst_seq, frag_seq, tx_counter = [int(x) for x in row[1:6]]
                tx_power = float(row[6])
                ac = row[7]
                is_data, is_burst, burst_rx, expired, acked = [bool(int(x)) for x in row[8:13]]
                app_tx_time, app_rx_time, enqueue_time, l2_rx_time = [float(x)*1e-3 for x in row[15:19]]

                packet_map[uid] = InFlightPacketInfo(
                    uid, size, tx_counter, tx_power, [phy_tx_time],
                    port, burst_seq, frag_seq, ac, 
                    is_data, is_burst, burst_rx, expired, acked,
                    app_tx_time, enqueue_time, l2_rx_time, app_rx_time)
            else:
                assert phy_tx_time[0] > packet_map[uid].phy_tx_time[-1][0], "phy tx time not increasing"
                packet_map[uid].phy_tx_time.append(phy_tx_time)
    
    return packet_map

def plot_bss_l2_thpt(bss_packet_map, ax):
    time_slot = 0.001   # in seconds
    window_size = 500   # number of slots
    timeline = np.arange(0, 6, time_slot)
    tot_thpt = np.zeros(len(timeline))
    node_thpt = {}
    for node_id, packet_map in bss_packet_map.items():
        node_thpt[node_id] = np.zeros(len(timeline))
        for _, info in packet_map.items():
            if info.is_burst == False or info.l2_rx_time == 0:
                continue
            left = np.searchsorted(timeline, info.l2_rx_time, side="left")
            right = np.searchsorted(timeline, info.l2_rx_time + time_slot*window_size, side="right")
            node_thpt[node_id][left:right] += info.size
        
        tot_thpt += node_thpt[node_id]
    
    if sum(tot_thpt) == 0:  # no traffic
        return 0
    
    for node_id, thpt in node_thpt.items():
        if node_id == 0:
            continue
        ax.plot(timeline, thpt*8*1e-6 / (time_slot*window_size), linewidth=1, label=f"STA-{node_id}")

    ax.plot(timeline, tot_thpt*8*1e-6 / (time_slot*window_size), linewidth=1, label="Total")
    

def plot_figure6():
    n_stas = list(range(2,11))
    beb = [11.36, 18.63, 25.4925, 29.394, 33.1667, 35.7414, 37.30375, 39.3633, 39.912]

    plt.figure()
    plt.plot(n_stas, beb, 'g-', linewidth=1, label='BEB (all legacy)')
    plt.legend()
    # plt.grid()
    plt.xlabel("Number of STAs")
    plt.ylabel("Collision Rate (\%)")
    plt.tight_layout()
    plt.savefig(path + "/fig6.png")
    plt.close()


if __name__ == '__main__':
    result_dir_list = [os.path.join(path, "../result-1744504679_lr1e-4"),]
    plot_figure4(result_dir_list)

    result_dir1 = os.path.join(path, "../result-1735735302-equalBEB/episode_0/")
    result_dir2 = os.path.join(path, "../result-1744544888-fig5/episode_0/")
    plot_figure5(result_dir1, result_dir2)

    plot_figure6()