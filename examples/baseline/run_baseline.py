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

sys.path.append(panlink_dir + '/contrib/ai/python_utils')
from ns3ai_utils import Experiment

sys.path.append(panlink_dir + '/env')
import cpp_py_interface as py_binding

sys.path.append(panlink_dir + '/agent')
from agent_manager import AgentManager

config_dir = os.path.join(panlink_dir, 'examples/configs')

if __name__ == '__main__':

    timestamp = str(int(time.time()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=11, help="random seed")
    parser.add_argument('--ns3_seed', type=int,
                        default=11, help='random seed for ns-3')
    parser.add_argument('--duration', type=float,
                        default=5, help='duration of the experiment')
    parser.add_argument('--use_rl', action='store_true',
                        help='whether to use rl')
    parser.add_argument('--use_intf', action='store_true',
                        help='whether to enable interference')
    parser.add_argument('--pcap', action='store_true',
                        help='whether to enable pcap')
    parser.add_argument('--stats', action='store_true',
                        help='whether to store results')
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # register agents
    agent_manager = AgentManager()
    n_vec_size = agent_manager.register_agents(os.path.join(config_dir, 'config_panlink.json'))
    
    # create and save ns3 settings
    ns3_settings = {
        'timestamp': timestamp,
        'seed': args.ns3_seed,
        'simulationTime': args.duration,
        'enableIntf': args.use_intf,
        'pcap': args.pcap,
        'stats': args.stats,
        "verbose": False
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

    try:
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

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception occurred: {}".format(e))
        print("Traceback:")
        traceback.print_tb(exc_traceback)
        exit(1)

    else:
        pass

    finally:
        time_cost = int(time.time()) - int(timestamp)
        print("Finally exiting...")
        print("Time cost: {}min {}s, Python time cost: {}s".format(
            time_cost // 60, time_cost % 60, py_time_cost))
        del exp