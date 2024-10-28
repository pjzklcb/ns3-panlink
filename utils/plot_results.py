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

class PlotResults(object):
    def __init__(self, timestamp, target_nodes = None):
        self.result_dir = "contrib/panlink/result-" + timestamp + "/"
        if not os.path.exists(self.result_dir):
            raise ValueError("Result directory {} does not exist".format(self.result_dir))
        if not os.path.exists(self.result_dir + "figures/"):
            os.makedirs(self.result_dir + "figures/")

        self.target_nodes = target_nodes
        self.n_nodes = 0    # number of nodes
        self.sim_time = 0   # simulation time
        self.bss_map = {}   # {ap_id: {sta_id, ...}}
        self.src_nodes = set()  # nodes with traffic

        self.timeline = {}  # {node_id: {port: [app_tx_time, ...]}}
        
    def do_plot(self):
        self.plot_position()
        self.read_csv_files()
        plt.figure()
        plt.close()
    
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

    def read_cw_from_csv(self):
        with open(self.result_dir+'csv/cw.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)

    def read_backoff_from_csv(self):
        with open(self.result_dir+'csv/backoff.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
    

if __name__ == '__main__':
    timestamp = "1730079312"
    PlotResults(timestamp).do_plot()