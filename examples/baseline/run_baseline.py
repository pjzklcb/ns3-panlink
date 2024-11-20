# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import sys
import json
import argparse
import time
import torch
import numpy as np
import traceback

ns3_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
panlink_dir = os.path.join(ns3_dir, 'contrib/panlink')
sys.path.append(panlink_dir + '/agent')
sys.path.append(panlink_dir + '/contrib/ai/python_utils')
sys.path.append(panlink_dir + '/env')
sys.path.append(panlink_dir + '/utils')

from ns3ai_utils import Experiment
from agent_manager import AgentManager
from plot_results import PlotResults
import cpp_py_interface as py_binding

config_dir = os.path.join(panlink_dir, 'configs')

if __name__ == '__main__':

    timestamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=11, help="random seed")
    parser.add_argument('--ns3_seed', type=int,
                        default=11, help='random seed for ns-3')
    parser.add_argument('--config_panlink', type=str,
                        default='panlink_no_agent.json', help='config file for panlink network')
    parser.add_argument('--config_intf', type=str,
                        default='intf_test1.json', help='config file for interference')
    parser.add_argument('--simulation_time', type=float,
                        default=5, help='total simulation time of ns-3')
    parser.add_argument('--enable_intf', action='store_true',
                        help='whether to enable interference')
    parser.add_argument('--pcap', action='store_true',
                        help='whether to enable pcap')
    parser.add_argument('--stats', action='store_true',
                        help='whether to store results')
    parser.add_argument('--ns3_log', action='store_true',
                        help='whether to print ns-3 log')
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # register agents
    agent_manager = AgentManager()
    n_vec_size = agent_manager.register_agents(os.path.join(config_dir, args.config_panlink))
    
    # create and save ns3 settings
    ns3_settings = {
        'timestamp': timestamp,
        'seed': args.ns3_seed,
        'configPanlink': os.path.join(config_dir, args.config_panlink),
        'configIntf': os.path.join(config_dir, args.config_intf),
        'simulationTime': args.simulation_time,
        'enableIntf': args.enable_intf,
        'pcap': args.pcap,
        'stats': args.stats,
        "verbose": args.ns3_log
    }

    # create result directory
    if args.stats:
        result_dir = os.path.join(panlink_dir, 'result-' + timestamp)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        with open(os.path.join(result_dir, 'ns3_settings.json'), 'w') as f:
            json.dump(ns3_settings, f, indent=4)

    exp = Experiment("panlink_baseline", ns3_dir, py_binding, 
                     handleFinish=True, useVector=True, vectorSize=n_vec_size,
                     segName = timestamp + ' Seg',
                     cpp2pyMsgName = timestamp + ' Cpp to Python Msg',
                     py2cppMsgName = timestamp + ' Python to Cpp Msg',
                     lockableName = timestamp + ' Lockable')
    msg_interface = exp.run(setting=ns3_settings, show_output=True)

    step = 0
    py_time_cost = 0

    while True:
        msg_interface.PyRecvBegin()
        if msg_interface.PyGetFinished():
            print("Simulation ended at {}".format(int(time.time())))
            break

        print("Step {}".format(step))
        step += 1

        tic = time.time()
        agent_manager.get_obs(msg_interface)
        py_time_cost += time.time() - tic
        msg_interface.PyRecvEnd()

        msg_interface.PySendBegin()
        agent_manager.set_action(msg_interface)
        msg_interface.PySendEnd()

    if args.stats:
        learn_dir = os.path.join(result_dir, 'learn')
        if not os.path.exists(learn_dir):
            os.makedirs(learn_dir)
        agent_manager.plot_results(learn_dir)
        agent_manager.save_models(learn_dir)

    time_cost = int(time.time()) - int(timestamp)
    print("Finally exiting...")
    print("Time cost: {}min {}s, Python time cost: {}s".format(
        time_cost // 60, time_cost % 60, py_time_cost))
    PlotResults(timestamp, target_nodes=[0, 1, 2, 3, 4]).do_plot()
    del exp