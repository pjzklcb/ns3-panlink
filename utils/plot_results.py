# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class PlotResults(object):
    def __init__(self, timestamp):
        self.result_dir = "contrib/panlink/result-" + timestamp + "/"
        if not os.path.exists(self.result_dir):
            raise ValueError("Result directory {} does not exist".format(self.result_dir))

        self.n_nodes = 0
        
    def do_plot(self):
        self.read_csv_files()
        plt.figure()
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
    timestamp = "1729673488"
    PlotResults(timestamp).do_plot()