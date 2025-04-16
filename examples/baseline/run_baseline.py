# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import os
import sys
import json
import argparse
import traceback
import time
import torch
import numpy as np
import random

ns3_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + '/../../../..')
panlink_dir = os.path.join(ns3_dir, 'contrib/panlink')
sys.path.append(panlink_dir + '/agent')
sys.path.append(panlink_dir + '/contrib/ai/python_utils')
sys.path.append(panlink_dir + '/env')
sys.path.append(panlink_dir + '/utils')

from ns3ai_utils import Experiment
from agent_manager import AgentManager
from plot_results import PlotResults
from config_generator import ConfigGenerator
import cpp_py_interface as py_binding

config_dir = os.path.join(panlink_dir, 'configs')
STATS_RECORD_STEP = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def run_new_episode(exp, ns3_settings, agent_manager, episode=0):
    ns3_time_cost = 0

    tic = time.time()
    msg_interface = exp.run(setting=ns3_settings, show_output=True)

    step = 0
    # try:
    while True:
        msg_interface.PyRecvBegin()
        ns3_time_cost += time.time() - tic  # wait for CppSendEnd
        if msg_interface.PyGetFinished():
            msg_interface.PyRecvEnd()
            break

        step += 1

        agent_manager.get_active_agent(msg_interface)
        if step % 100 == 0:
            print(f"Episode {episode}, step {step}, sim time {agent_manager.active_agent.sim_time:.2f}")
        while True:
            agent_manager.get_obs(msg_interface)
            msg_interface.PyRecvEnd()
            if agent_manager.has_more_data():
                msg_interface.PyRecvBegin()
            else:
                break

        msg_interface.PySendBegin()
        agent_manager.set_action(msg_interface)
        tic = time.time()   # reset tic before CppRecvEnd
        msg_interface.PySendEnd()

    # except Exception as e:
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     print("Episode {} finished with exception {}".format(episode, e))
    #     print("Traceback (most recent call last):")
    #     traceback.print_tb(exc_traceback)
    
    exp.kill()
    print("Simulation ended at {}".format(int(time.time())))
    msg_interface.PySetRestart()

    return ns3_time_cost


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=11, help="random seed")
    parser.add_argument('--n_episodes', type=int,
                        default=1, help='number of episodes')
    parser.add_argument('--marl', action='store_true',
                        help='whether to use MARL')
    parser.add_argument('--config_panlink', type=str,
                        default='panlink_no_agent.json', help='config file for panlink network')
    parser.add_argument('--config_intf', type=str,
                        default='intf_test1.json', help='config file for interference')
    parser.add_argument('--simulation_time', type=float,
                        default=5, help='total simulation time of ns-3')
    parser.add_argument('--enable_intf', action='store_true',
                        help='whether to enable interference')
    parser.add_argument('--max_mpdus', type=int,
                        default=255, help='maximum number of MPDUs per AMPDU')
    parser.add_argument('--pcap', action='store_true',
                        help='whether to enable pcap')
    parser.add_argument('--stats', action='store_true',
                        help='whether to store results')
    parser.add_argument('--ns3_log', action='store_true',
                        help='whether to print ns-3 log')
    args = parser.parse_args()
    
    # set seed
    set_seed(args.seed)
    
    # create result directory
    timestamp = str(int(time.time()))
    if args.stats:
        result_dir = os.path.join(panlink_dir, 'result-' + timestamp)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    # initialize config file
    config_generator = ConfigGenerator(os.path.join(config_dir, args.config_panlink))
    config_panlink = config_generator.generate(
        os.path.join(result_dir, "config_panlink.json"), 0, args.n_episodes, init=True)
    
    # register agents
    agent_manager = AgentManager(args.marl)
    n_vec_size = agent_manager.register_agents(config_panlink)

    # create interprocess interface
    exp = Experiment("panlink_baseline", ns3_dir, py_binding, 
                    handleFinish=True, useVector=True, vectorSize=n_vec_size, shmSize=65536,
                    segName = timestamp + ' Seg',
                    cpp2pyMsgName = timestamp + ' Cpp to Python Msg',
                    py2cppMsgName = timestamp + ' Python to Cpp Msg',
                    lockableName = timestamp + ' Lockable')
    
    ns3_time_cost = 0
    for i_episode in range(args.n_episodes):
        print("Episode {} starts".format(i_episode))

        # generate variable config file
        if config_generator.variable:
            generate_config_dir = os.path.join(result_dir, 'configs')
            if not os.path.exists(generate_config_dir):
                os.makedirs(generate_config_dir)
            config_panlink = config_generator.generate(
                os.path.join(generate_config_dir, f"config_panlink_{i_episode}.json"), i_episode, args.n_episodes)

        # create and save ns3 settings
        ns3_settings = {
            'timestamp': timestamp,
            'streamNumber': args.seed,
            'configPanlink': config_panlink,
            'configIntf': os.path.join(config_dir, args.config_intf),
            'simulationTime': args.simulation_time,
            'enableIntf': args.enable_intf,
            'maxMpdus': args.max_mpdus,
            'pcap': args.pcap,
            'stats': args.stats,
            "verbose": args.ns3_log
        }

        if args.stats:
            if i_episode % STATS_RECORD_STEP == 0 or i_episode == args.n_episodes - 1:
                episode_dir = os.path.join(result_dir, 'episode_' + str(i_episode))
                os.makedirs(episode_dir)
                learn_dir = os.path.join(episode_dir, 'learn')
                os.makedirs(learn_dir)
                agent_manager.set_save_dir(learn_dir)

                ns3_settings['resultDir'] = episode_dir
                ns3_settings['stats'] = True
                with open(os.path.join(episode_dir, 'ns3_settings.json'), 'w') as f:
                    json.dump(ns3_settings, f, indent=4)
            else:
                agent_manager.set_save_dir(None)
                ns3_settings['stats'] = False
        
        # run ns-3 simulation
        ns3_time_cost += run_new_episode(exp, ns3_settings, agent_manager, episode=i_episode)

        if args.stats:
            if i_episode % STATS_RECORD_STEP == 0 or i_episode == args.n_episodes - 1:
                agent_manager.plot_results()
                agent_manager.save_models()
        
        # reset env. Training if model is on-policy
        agent_manager.reset()
        if args.stats:
            agent_manager.plot_episodes(result_dir)

    del exp
    time_cost = int(time.time()) - int(timestamp)
    print("Finally exiting...")
    print("Time cost: {}min {}s, ns-3 time cost: {}min {:.2f}s".format(
        time_cost//60, time_cost%60, int(ns3_time_cost)//60, ns3_time_cost%60))
    
    if args.stats:
        PlotResults(timestamp, args.n_episodes-1, target_nodes=[0,1,2,3,4]).do_plot()