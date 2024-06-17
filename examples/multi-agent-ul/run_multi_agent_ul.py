import os 
import sys
ns3_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
panlink_path = os.path.abspath(ns3_path + '/contrib/panlink')
sys.path.append(panlink_path + '/env')
sys.path.append(panlink_path + '/agent')

import argparse
import traceback
import time
import panlink_py_interface as py_binding
from channel_access_rl_agent import ChannelAccessRlAgent
from ns3ai_utils import Experiment

if __name__ == '__main__':
    timestamp = int(time.time())

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        help='set seed for reproducibility')
    parser.add_argument('--sim_seed', type=int,
                        help='set simulation run number')
    parser.add_argument('--duration', type=float, default=5,
                        help='set simulation duration (seconds)')
    parser.add_argument('--show_log', action='store_true',
                        help='whether show observation and action')
    parser.add_argument('--result', action='store_true',
                        help='whether output figures')
    parser.add_argument('--result_dir', type=str, default='./rl_tcp_results', 
                        help='output figures path')
    parser.add_argument('--use_rl', action='store_true',
                        help='whether use rl algorithm')

    args = parser.parse_args()

    ns3Settings = {
        'pktSize': 1500,
        'duration': args.duration,
        'gi': 800,
        'channelWidth': 20,
        'rng': 2,
        'apNodes': 4,
        'networkSize': 4,
        'ring': 0,
        'maxMpdus': 5,
        'prop': 'log',
        'app': 'setup-done',
        'pktInterval': 5000,
        'boxsize': 25,
        'drl': True,
        'configFile': ns3_path+'/contrib/panlink/examples/configs/config.txt',
    }
    n_ap = int(ns3Settings['apNodes'])
    n_sta = int(ns3Settings['networkSize'])
    n_total = n_ap * (n_sta + 1)

    # Get number of actions from gym action space
    n_actions = -62 - (-82) + 1
    n_obs = (n_sta + 1) * (n_total + 1)

    exp = Experiment("ns3channel_access", ns3_path, py_binding,
                    handleFinish=True, useVector=True, vectorSize=n_total)
    msg_interface = exp.run(setting=ns3Settings, show_output=True)

    agent = ChannelAccessRlAgent(n_obs, n_actions, n_total, n_ap, n_sta, show_log=args.show_log)

    try:
        while True:
            # Get current state from C++
            msg_interface.PyRecvBegin()
            if msg_interface.PyGetFinished():
                print("Finished")
                break
            obs = agent.choose_action(msg_interface)
            msg_interface.PyRecvEnd()

            # RL algorithm learn here
            agent.learn()

            # put the action back to C++
            msg_interface.PySendBegin()
            agent.set_action(msg_interface)
            msg_interface.PySendEnd()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("Exception occurred: {}".format(e))
        print("Traceback:")
        traceback.print_tb(exc_traceback)
        exit(1)

    else:
        agent.plot_results()

    finally:
        print("Finally exiting...")
        del exp
