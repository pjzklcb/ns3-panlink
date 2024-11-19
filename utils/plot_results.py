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
            self.sim_time = float(json.load(f)["simulationTime"]) + 1.0

        self.target_nodes = target_nodes
        self.n_nodes = 0    # number of nodes
        self.bss_map = {}   # {ap_id: {sta_id, ...}}
        self.channel_load = set()  # set((phy_start, end, node_id)) avoid AMPDU duplication

    def do_plot(self):
        self.plot_position()
        self.process_csv_files()
        self.plot_channel_load([[3.0,3.5], [4.6,4.7]])
    
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
                if node_type == "PanlinkAp":
                    self.bss_map[bss_id] = []
                    ap_pos[node_id] = [x, y]
                    plt.plot(x, y, 'ro', markersize=4, label="" if n_po else "Panlink AP")
                    n_po += 1
                elif node_type == "PanlinkSta":
                    self.bss_map[bss_id].append(node_id)
                    plt.plot(x, y, 'r*', markersize=8, label="" if n_pc else "Panlink STA")
                    n_pc += 1
                    plt.plot([ap_pos[bss_id][0], x], [ap_pos[bss_id][1], y], 'k--')
                elif node_type == "Ap":
                    self.bss_map[bss_id] = []
                    ap_pos[node_id] = [x, y]
                    plt.plot(x, y, 'bo', markersize=4, label="" if n_ap else "AP")
                    n_ap += 1
                elif node_type == "Sta":
                    self.bss_map[bss_id].append(node_id)
                    plt.plot(x, y, 'gx', markersize=4, label="" if n_sta else "STA")
                    n_sta += 1
                    plt.plot([ap_pos[bss_id][0], x], [ap_pos[bss_id][1], y], 'k--')
                elif node_type == "Probe":
                    plt.plot(x, y, 'mx', markersize=4, alpha=0.5, label="" if n_probe else "Probe")
                    n_probe += 1
                else:
                    print("Node type error.")
                    exit(1)

                self.n_nodes += 1
        
        plt.legend(loc="upper left", ncol=4)
        plt.xlabel("x-coordinate (m)")
        plt.xlabel("y-coordinate (m)")
        plt.title("Node position")
        plt.savefig(self.result_dir + "figures/topology.png", bbox_inches='tight', dpi=250)
        plt.close()

    def process_csv_files(self):
        csv_files = [f for f in os.listdir(self.result_dir+'csv/') if f.endswith('.csv')]
        
        # read MAC csv files
        if "cw.csv" in csv_files:
            self.read_cw_from_csv()   
        if "backoff.csv" in csv_files:
            self.read_backoff_from_csv()
        try:
            self.plot_cw_backoff()
        except:
            print("No CW and backoff data found.")

        # read node_*_packet.csv based on bss map
        for ap_id, sta_list in self.bss_map.items():
            bss_packet_map = {}

            csv_file = "node_" + str(ap_id) + "_packet.csv"
            assert csv_file in csv_files, "CSV file {} not found".format(csv_file)
            bss_packet_map[ap_id] = self.read_packet_map_from_csv(csv_file, ap_id)
            
            for sta_id in sta_list:
                csv_file = "node_" + str(sta_id) + "_packet.csv"
                assert csv_file in csv_files, "CSV file {} not found".format(csv_file)
                bss_packet_map[sta_id] = self.read_packet_map_from_csv(csv_file, sta_id)

            # plot l2 thpt
            tot_thpt = self.plot_bss_l2_thpt(bss_packet_map)
            # plot delay
            if tot_thpt > 0:
                self.process_delay(bss_packet_map)

    def read_cw_from_csv(self):
        self.cw_map = {} # {node_id: {"AC BE": [[time, cw], ...],},}
        with open(self.result_dir+'csv/cw.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)    # skip header
            for row in reader:
                node_id, ac, time, cw = int(row[0]), row[1], float(row[2])*1e-3, int(row[3])

                if node_id not in self.cw_map:
                    self.cw_map[node_id] = {}
                if ac not in self.cw_map[node_id]:
                    self.cw_map[node_id][ac] = []
                self.cw_map[node_id][ac].append([time, cw])

    def read_backoff_from_csv(self):
        self.backoff_map = {} # {node_id: {"AC BE": [[time, backoff], ...],},}
        with open(self.result_dir+'csv/backoff.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                node_id, ac, time, backoff = int(row[0]), row[1], float(row[2])*1e-3, int(row[3])
                backoff = 0.5 if backoff == 0 else backoff  # set to 0.5 to be shown in log scale

                if node_id not in self.backoff_map:
                    self.backoff_map[node_id] = {}
                if ac not in self.backoff_map[node_id]:
                    self.backoff_map[node_id][ac] = []
                self.backoff_map[node_id][ac].append([time, backoff])
    
    def plot_cw_backoff(self):
        # plot cw and backoff of target nodes
        node_set = self.target_nodes if self.target_nodes else [k for k in self.cw_map.keys()]
        for node_id in node_set:
            if node_id not in self.cw_map or node_id not in self.backoff_map:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(8, 6))
            for ac in self.cw_map[node_id]:
                if ac == "AC BE":
                    subfig = axs[0, 0]
                    subfig.set_title("AC BE")
                    subfig.set_ylim([0.4, 1100])
                elif ac == "AC BK":
                    subfig = axs[0, 1]
                    subfig.set_title("AC BK")
                    subfig.set_ylim([0.4, 1100])
                elif ac == "AC VO":
                    subfig = axs[1, 0]
                    subfig.set_title("AC VO")
                    subfig.set_ylim([0.4, 18])
                elif ac == "AC VI":
                    subfig = axs[1, 1]
                    subfig.set_title("AC VI")
                    subfig.set_ylim([0.4, 35])
                else:
                    continue
                
                prev = 0
                for time, cw in self.cw_map[node_id][ac]:
                    subfig.plot([prev, time], [cw, cw], "b")
                    prev = time
                subfig.plot([prev, self.sim_time], [cw, cw], "b")

                time_list = []
                backoff_list = []
                for time, backoff in self.backoff_map[node_id][ac]:
                    time_list.append(time)
                    backoff_list.append(backoff)
                subfig.scatter(time_list, backoff_list, color="r", marker='.', s=8, alpha=0.5,
                               label='average backoff={:.2f}'.format(np.mean(backoff_list)))
                
                subfig.legend(loc='upper right')
                subfig.set_yscale('log', base=2)
                subfig.set_xlabel("Time (s)")
                subfig.set_ylabel("CW")
            
            fig.tight_layout()
            fig.savefig(self.result_dir + "figures/node_" + str(node_id) + "_cw_backoff.png")
            plt.close(fig)

    def read_packet_map_from_csv(self, csv_file, node_id):
        print("Processing packet map for node_{}".format(node_id))
        packet_map = {}   # packet map for node_id {uid: InFlighPacketInfo, ...}
        with open(self.result_dir+'csv/'+csv_file, 'r') as f:
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
                
                if phy_tx_time[0] != 0 and phy_tx_time[1] != 0:
                    self.channel_load.add(tuple(phy_tx_time) + (node_id,))
            
        return packet_map
    
    def plot_bss_l2_thpt(self, bss_packet_map):
        time_slot = 0.001   # in seconds
        window_size = 500   # number of slots
        timeline = np.arange(0, self.sim_time, time_slot)
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
        
        plt.figure()
        for node_id, thpt in node_thpt.items():
            plt.plot(timeline, thpt*8*1e-6 / (time_slot*window_size), linewidth=1, label="Node-{}".format(node_id))

        plt.plot(timeline, tot_thpt*8*1e-6 / (time_slot*window_size), linewidth=1, label="Total")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (Mbps)")
        plt.legend()
        plt.savefig(self.result_dir + "figures/bss_" + str(list(bss_packet_map.keys())[0]) + "_thpt.png")
        plt.close()
        return sum(tot_thpt)

    def process_delay(self, bss_packet_map):
        for node_id, packet_map in bss_packet_map.items():
            # l2 delay
            l2_delay = {}   # {port: [(tx_time, delay), ...],}#
            drop_before_enqueue = {}    # {port: [tx_time, ...],}
            expired = {}
            dropped = {}
            # e2e delay
            e2e_delay = {}  # {port: {burst_seq:(tx_time, delay), ...},}
            incomplete = {} # {port: {burst_seq:tx_time, ...},}
            for _, info in packet_map.items():
                if info.is_burst == False:
                    continue
                # calculate l2 delay
                if info.enqueue_time == 0:  # not enqueued
                    if info.port not in drop_before_enqueue:
                        drop_before_enqueue[info.port] = []
                    drop_before_enqueue[info.port].append(info.app_tx_time)
                elif info.tx_counter == 0:  # not transmitted
                    if info.port not in expired:
                        expired[info.port] = []
                    expired[info.port].append(info.app_tx_time)
                elif info.l2_rx_time == 0:  # not received
                    if info.port not in dropped:
                        dropped[info.port] = []
                    dropped[info.port].append(info.app_tx_time)
                else:
                    # packet may rx successfully but simulation ends before acked
                    if info.port not in l2_delay:
                        l2_delay[info.port] = []
                    l2_delay[info.port].append((info.app_tx_time, (info.l2_rx_time-info.app_tx_time) * 1e3))

                # calculate e2e delay
                if info.burst_rx:   # if burst is received, find the max e2e delay
                    if info.port not in e2e_delay:
                        e2e_delay[info.port] = {}
                    if info.burst_seq not in e2e_delay[info.port]:
                        e2e_delay[info.port][info.burst_seq] = (info.app_tx_time, 
                                                                (info.app_rx_time-info.app_tx_time) * 1e3)
                    else:
                        delay = max(e2e_delay[info.port][info.burst_seq][1], 
                                    (info.app_rx_time-info.app_tx_time) * 1e3)
                        e2e_delay[info.port][info.burst_seq] = (info.app_tx_time, delay)
                else:
                    if info.port not in incomplete:
                        incomplete[info.port] = {}
                    incomplete[info.port][info.burst_seq] = info.app_tx_time
            
            # plot l2 delay
            self.plot_l2_delay(node_id, l2_delay, drop_before_enqueue, expired, dropped)

            # plot l2 delay cdf of tartgeted nodes
            if self.target_nodes is None or node_id in self.target_nodes:
                self.plot_l2_delay_cdf(node_id, l2_delay, dropped)

            # plot e2e delay of tartgeted nodes
            if self.target_nodes is None or node_id in self.target_nodes:
                self.plot_e2e_delay(node_id, e2e_delay, incomplete)
    
    def plot_l2_delay(self, node_id, l2_delay, drop_before_enqueue, expired, dropped):
        if len(l2_delay) == 0:
            return
        
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, port in enumerate(l2_delay):
            x = [t for t, _ in l2_delay[port]]
            y = [d for _, d in l2_delay[port]]
            plt.plot(x, y, color=colors[i % len(colors)], linestyle=':', linewidth=1, 
                        label="Port-{}".format(port))
        for i, port in enumerate(drop_before_enqueue):
            plt.scatter(drop_before_enqueue[port], np.zeros(len(drop_before_enqueue[port])), 
                        c='m', marker='+', alpha=0.5, label="Drop before enqueue" if i==0 else "")
        for i, port in enumerate(expired):
            plt.scatter(expired[port], np.zeros(len(expired[port])), 
                        c='k', marker='^', alpha=0.5, label="Expired" if i==0 else "")
        for i, port in enumerate(dropped):
            plt.scatter(dropped[port], np.zeros(len(dropped[port])), 
                        c='r', marker='x', alpha=0.5, label="Dropped" if i==0 else "")

        plt.xlim([0, self.sim_time])
        plt.xlabel("Time (s)")
        plt.ylabel("L2 Delay (ms)")
        plt.legend()
        plt.savefig(self.result_dir + "figures/node_" + str(node_id) + "_l2_delay.png")
        plt.close()

    def plot_l2_delay_cdf(self, node_id, l2_delay, dropped):
        tail_latency = 30 # ms
        delay_list = []
        n_drops = 0
        for port, delay in l2_delay.items():
            if len(delay) < 10: # ignore app with too few successful packets
                continue
            delay_list += [d for _, d in delay]
            if port in dropped:
                n_drops += len(dropped[port])
        
        if len(delay_list) == 0:
            return
        
        plt.figure()
        x = np.sort(delay_list)
        y = np.arange(1, len(x)+1) / len(x)
        plt.plot(x, y, linewidth=1, label="L2 Delay")
        plt.plot(0, 0, 'w', label='mean = {:.2f}'.format(np.mean(delay_list)))
        plt.plot(0, 0, 'w', label='std = {:.2f}'.format(np.std(delay_list)))

        drop_rate = n_drops / (n_drops + len(delay_list))
        prob_tail_latency = np.sum(np.array(delay_list) > tail_latency) / len(delay_list)
        plt.plot(0, 0, 'w', label='drop rate = {:.2f}%'.format(drop_rate*100))
        plt.plot(0, 0, 'w', label='$P_{tail}$'+' = {:.2f}%'.format(prob_tail_latency*100))
        plt.axhline(1-prob_tail_latency, color='r', linestyle=':', linewidth=1)
        plt.axvline(tail_latency, color='r', linestyle=':', linewidth=1)

        plt.xlabel("L2 Delay (ms)")
        plt.ylabel("CDF")
        plt.legend()
        plt.savefig(self.result_dir + "figures/node_" + str(node_id) + "_l2_delay_cdf.png")
        plt.close()
        
    def plot_e2e_delay(self, node_id, e2e_delay, incomplete):
        if len(e2e_delay) == 0:
            return
        
        plt.figure()
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        for i, port in enumerate(e2e_delay):
            x = [t for t, _ in e2e_delay[port].values()]
            y = [d for _, d in e2e_delay[port].values()]
            plt.plot(x, y, color=colors[i % len(colors)], linestyle=':', linewidth=1, 
                     label="Port-{}".format(port))
        
        for i, port in enumerate(incomplete):
            plt.scatter(incomplete[port].values(), np.zeros(len(incomplete[port].values())), 
                        c='r', marker='x', alpha=0.5, label="Incomplete" if i==0 else "")
            
        plt.xlim([0, self.sim_time])
        plt.xlabel("Time (s)")
        plt.ylabel("E2E Delay (ms)")
        plt.legend()
        plt.savefig(self.result_dir + "figures/node_" + str(node_id) + "_e2e_delay.png")
        plt.close()

    # @todo: need to consider hidden nodes
    def plot_channel_load(self, intervals = [[3.0, 3.5]]):
        events = list(self.channel_load) + [(self.sim_time, self.sim_time, 0)] # add a dummy event at end
        events.sort(key=lambda x: x[0])  # sort by start time

        idle_time = 0
        util_time = 0
        n_collisions = {id: 0 for id in self.target_nodes}
        n_transmits = {id: 0 for id in self.target_nodes}
        prev_start, prev_end = -1, 0
        current_events = []
        for event in events:
            start, end, node_id = event
            if node_id in n_transmits:
                n_transmits[node_id] += 1
            if prev_start <= start < prev_end:
                # if overlapping
                current_events.append(event)
                prev_end = max(prev_end, end)
            else:
                idle_time += start - prev_end
                # handle the previous collision
                if len(current_events) == 1:
                    util_time += prev_end - max(0, prev_start)
                elif len(current_events) > 1:
                    collision_nodes = set()
                    for s, e, id in current_events:
                        # if id in collision_nodes:
                        #     print("Warning: node-{} may be a hidden node to {}".format(id, collision_nodes))
                        collision_nodes.add(id)
                        if id in n_collisions:
                            n_collisions[id] += 1
                current_events = [event]
                prev_start, prev_end = start, end

        idle_rate = round(idle_time*100 / self.sim_time, 2)
        util_rate = round(util_time*100 / self.sim_time, 2)
        print('idle rate = {}%'.format(idle_rate))
        print('util rate = {}%'.format(util_rate))
        for id, n in n_collisions.items():
            collision_rate = round(n*100 / n_transmits[id], 2)
            print('node-{} transmits {} times, collides {} times, collision rate={}%'.format(
                id, n_transmits[id], n, collision_rate))
        # write to txt
        with open(self.result_dir + "figures/channel_load.txt", 'w') as f:
            f.write('idle rate = {}%\n'.format(idle_rate))
            f.write('util rate = {}%\n'.format(util_rate))
            for id, n in n_collisions.items():
                f.write('node-{} transmits {} times, collides {} times, collision rate={}%'.format(
                id, n_transmits[id], n, collision_rate))

        # plot channel load in each interval
        for left, right in intervals:
            idle_time = 0
            util_time = 0
            n_collisions = 0

            plt.figure()
            prev_start, prev_end = -1, left
            current_events = []
            for event in events:
                start, end, _ = event
                if end < left:
                    continue

                if prev_start <= start < prev_end:
                    # if overlapping
                    current_events.append(event)
                    prev_end = max(prev_end, end)
                else:
                    idle_time += max(min(start, right), left) - min(prev_end, right)
                    # handle the previous collision
                    if len(current_events) == 1:
                        util_time += min(prev_end, right) - max(prev_start, left)
                        prev_id = current_events[0][2]
                        plt.plot([max(prev_start, left), min(prev_end, right)], [prev_id, prev_id], 'b', linewidth=2.0)
                    elif len(current_events) > 1:
                        n_collisions += 1
                        plt.plot([max(prev_start, left), max(prev_start, left)], [0, self.n_nodes], 'g', alpha=0.5)
                        for s, e, id in current_events:
                            plt.plot([max(s, left), min(e, right)], [id, id], 'm', linewidth=2.0)
                    prev_start, prev_end = start, end
                    current_events = [event]
                if start > right:
                    break

            idle_rate = round(idle_time*100 / (right - left), 2)
            util_rate = round(util_time*100 / (right - left), 2)
            plt.plot(left, 0, 'w', label='idle rate = {}%'.format(idle_rate))
            plt.plot(left, 0, 'w', label='util rate = {}%'.format(util_rate))
            plt.plot(left, 0, 'w', label='# of collisions = {}'.format(n_collisions))
            print("Interval [{}, {}]: idle rate = {}%, util rate = {}%, collisions = {}".format(
                left, right, idle_rate, util_rate, n_collisions))
            plt.xlabel("Time (s)")
            plt.ylabel("Node ID")
            plt.legend(loc='upper right')
            plt.savefig(self.result_dir + "figures/channel_load_{}_{}.png".format(left, right), 
                        bbox_inches='tight', dpi=250)
            plt.close()
        

if __name__ == '__main__':
    timestamp = "1731998150"
    PlotResults(timestamp, target_nodes=[0,1,2,3,4]).do_plot()