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
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create result directory
    result_dir = os.path.join(ns3_dir, 'result-'+timestamp)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    # create and save ns3 settings
    ns3_settings = {
        'timestamp': timestamp,
        'seed': args.ns3_seed,
        'simulationTime': args.duration,
        'enableIntf': args.use_intf,
        'pcap': args.pcap
    }
    with open(os.path.join(result_dir, 'ns3_settings.json'), 'w') as f:
        json.dump(ns3_settings, f, indent=4)

    exp = Experiment("panlink_baseline", ns3_dir, py_binding, 
                     handleFinish=True, useVector=True, vectorSize=1,
                     segName = timestamp + ' Seg',
                     cpp2pyMsgName = timestamp + ' Cpp to Python Msg',
                     py2cppMsgName = timestamp + ' Python to Cpp Msg',
                     lockableName = timestamp + ' Lockable')
    msg_interface = exp.run(setting=ns3_settings, show_output=True)

    try:
        while True:
            msg_interface.PyRecvBegin()
            msg_interface.PySendBegin()
            if msg_interface.PyGetFinished():
                break
            msg_interface.GetPy2CppStruct().nss, msg_interface.GetPy2CppStruct().next_mcs = (
                get_action(msg_interface.GetCpp2PyStruct()))
            msg_interface.PyRecvEnd()
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
        print("Finally exiting...")
        del exp