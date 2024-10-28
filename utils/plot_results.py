# Copyright (c) 2024
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

class InFlightPacketInfo:
    def __init__(self, uid, size, tx_counter, tx_power, phy_tx_time,
                 port=0, burst_seq=0, frag_seq=0, ac="AC BE", 
                 is_data=False, is_burst=False, burst_rx=False, expired=False, acked=False, 
                 app_tx_time=0, enqueue_time=0, l2_rx_time=0, app_rx_time=0):
        self.uid = uid
        self.size = size
        self.tx_counter = tx_counter
        self.tx_power = tx_power
        self.port = port
        self.burst_seq = burst_seq
        self.frag_seq = frag_seq
        self.is_data = is_data
        self.is_burst = is_burst
        self.burst_rx = burst_rx
        self.expired = expired
        self.acked = acked
        self.ac = ac
        self.phy_tx_time = phy_tx_time
        self.app_tx_time = app_tx_time
        self.app_rx_time = app_rx_time
        self.enqueue_time = enqueue_time
        self.l2_rx_time = l2_rx_time


class PlotResults(object):
    def __init__(self, timestamp, target_nodes = None):
        self.result_dir = "contrib/panlink/result-" + timestamp + "/"
        if not os.path.exists(self.result_dir):
            raise ValueError("Result directory {} does not exist".format(self.result_dir))
        if not os.path.exists(self.result_dir + "figures/"):
            os.makedirs(self.result_dir + "figures/")

        with open(self.result_dir+'ns3_settings.json', 'r') as f:
            self.sim_time = float(json.load(f)["simulationTime"])

        self.target_nodes = target_nodes
        self.n_nodes = 0    # number of nodes
        self.bss_map = {}   # {ap_id: {sta_id, ...}}
        self.channel_load = {}  # {node_id: {(phy_begin, phy_end), ...}}
        
    def do_plot(self):
        self.plot_position()
        self.read_csv_files()
    
    def plot_position(self):
        plt.figure()

        with open(self.result_dir+'/topology.json', 'r') as f:
            doc = json.load(f)
            nodes_info = doc["nodes"]

            ap_pos = {}
            n_po, n_pc, n_ap, n_sta, n_probe = 0, 0, 0, 0, 0
            for node_info in nodes_info:
                bss_id = int(node_info["bssid"])
                node_id = int(node_info["id"])
                node_type = node_info["type"]
                x, y = float(node_info["loc_x"]), float(node_info["loc_y"])
                if node_type == "PO":
                    self.bss_map[bss_id] = []
                    ap_pos[node_id] = [x, y]
                    plt.plot(x, y, 'ro', markersize=4, label="" if n_po else "PO")
                    n_po += 1
                elif node_type == "PC":
                    self.bss_map[bss_id].append(node_id)
                    plt.plot(x, y, 'r*', markersize=8, label="" if n_pc else "PC")
                    n_pc += 1
                    plt.plot([ap_pos[bss_id][0], x], [ap_pos[bss_id][1], y], 'k--')
                elif node_type == "AP":
                    self.bss_map[bss_id] = []
                    ap_pos[node_id] = [x, y]
                    plt.plot(x, y, 'bo', markersize=4, label="" if n_ap else "AP")
                    n_ap += 1
                elif node_type == "STA":
                    self.bss_map[bss_id].append(node_id)
                    plt.plot(x, y, 'gx', markersize=4, label="" if n_sta else "STA")
                    n_sta += 1
                    plt.plot([ap_pos[bss_id][0], x], [ap_pos[bss_id][1], y], 'k--')
                elif node_type == "PROBE":
                    plt.plot(x, y, 'mx', markersize=4, alpha=0.5, label="" if n_probe else "PROBE")
                    n_probe += 1
                else:
                    print("Node type error.")
                    exit(1)

                self.n_nodes += 1
        
        plt.legend()
        plt.xlabel("x-coordinate (m)")
        plt.xlabel("y-coordinate (m)")
        plt.title("Node position")
        plt.savefig(self.result_dir + "figures/topology.png", bbox_inches='tight', dpi=250)
        plt.close()

    def read_csv_files(self):
        csv_files = [f for f in os.listdir(self.result_dir+'csv/') if f.endswith('.csv')]
        if "cw.csv" in csv_files:
            self.read_cw_from_csv()
        
        if "backoff.csv" in csv_files:
            self.read_backoff_from_csv()
        
        # read packet map from node*_packet.csv
        for csv_file in csv_files:
            if not csv_file.endswith("_packet.csv"):
                continue
            node_id = int(csv_file.split("_")[0][4:]) # get index from filename
            self.read_packet_map_from_csv(csv_file, node_id)

    def read_cw_from_csv(self):
        self.cw_map = {} # {node_id: {"AC BE": [cw, ...], ...}}
        with open(self.result_dir+'csv/cw.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_id, ac, cw = int(row[0]), row[1], int(row[2])

                if node_id not in self.cw_map:
                    self.cw_map[node_id] = {}
                if ac not in self.cw_map[node_id]:
                    self.cw_map[node_id][ac] = []
                self.cw_map[node_id][ac].append(cw)

    def read_backoff_from_csv(self):
        self.backoff_map = {} # {node_id: {"AC BE": [backoff, ...], ...}}
        with open(self.result_dir+'csv/backoff.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                node_id, ac, backoff = int(row[0]), row[1], int(row[2])

                if node_id not in self.backoff_map:
                    self.backoff_map[node_id] = {}
                if ac not in self.backoff_map[node_id]:
                    self.backoff_map[node_id][ac] = []
                self.backoff_map[node_id][ac].append(backoff)
    
    def read_packet_map_from_csv(self, csv_file, node_id):
        self.channel_load[node_id] = {}
        packet_map = {}   # packet map for node_id {uid: InFlighPacketInfo, ...}
        with open(self.result_dir+'csv/'+csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)    # skip header
            for row in reader:
                # parse packet info
                uid, size, port, burst_seq, frag_seq, tx_counter = [int(x) for x in row[0:6]]
                tx_power = float(row[6])
                ac = row[7]
                is_data, is_burst, burst_rx, expired, acked = [bool(x) for x in row[8:13]]
                phy_tx_time = [float(x) for x in row[13:15]]
                app_tx_time, app_rx_time, enqueue_time, l2_rx_time = [float(x) for x in row[15:19]]

                if uid not in packet_map:
                    phy_tx_time_list = [phy_tx_time]
                else:
                    assert phy_tx_time[0] > packet_map[uid].phy_tx_time[-1][0], "phy tx time not increasing"
                    phy_tx_time_list = packet_map[uid].phy_tx_time.append(phy_tx_time)

                packet_map[uid] = InFlightPacketInfo(
                    uid, size, tx_counter, tx_power, phy_tx_time_list,
                    port, burst_seq, frag_seq, ac, 
                    is_data, is_burst, burst_rx, expired, acked,
                    app_tx_time, enqueue_time, l2_rx_time, app_rx_time)


if __name__ == '__main__':
    timestamp = "1730079312"
    PlotResults(timestamp).do_plot()