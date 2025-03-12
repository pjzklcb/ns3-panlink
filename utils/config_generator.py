# Copyright (c) 2025
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

import json
import random
import math


class ConfigGenerator:
    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as f:
            config = json.load(f)
            if "variable" not in config:
                self.variable = None
            else:
                self.variable = config["variable"]
    
    def generate(self, output_file, init=False):
        if self.variable is None:
            return self.config_file
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            n_stas = config["n_stas"]
            n_panlink_stas = config["n_panlink_stas"]
            distance = config["distance"]

        # generate number of stas
        if init or n_stas["dist"] == "constant":
            num_stas = n_stas["max"]
        elif n_stas["dist"] == "uniform":
            num_stas = random.randint(n_stas["min"], n_stas["max"])
        else:
            raise ValueError(f"Unsupported distribution {n_stas['dist']} for n_stas")

        # generate number of panlink stas
        if init or n_panlink_stas["dist"] == "constant":
            num_panlink_stas = min(num_stas, n_panlink_stas["max"])
        elif n_panlink_stas["dist"] == "uniform":
            num_panlink_stas = random.randint(min(num_stas, n_panlink_stas["min"]), 
                                              min(num_stas, n_panlink_stas["max"]))
        else:
            raise ValueError(f"Unsupported distribution {n_panlink_stas['dist']} for n_panlink_stas")
        print(f"Generate total number of stas: {num_stas}, panlink stas: {num_panlink_stas}")

        # generate distance
        if distance["dist"] == "constant":
            r_list = [distance["max"]] * num_stas
        elif distance["dist"] == "uniform":
            r_list = [random.uniform(distance["min"], distance["max"]) for _ in range(num_stas)]
        else:
            raise ValueError(f"Unsupported distribution {distance['dist']} for distance")

        theta_list = [random.uniform(0, math.pi * 2) for _ in range(num_stas)]
        x_list = [r * math.cos(theta) for r, theta in zip(r_list, theta_list)]
        y_list = [r * math.sin(theta) for r, theta in zip(r_list, theta_list)]

        # generate ap
        ap_node = {
            "id": 0,
            "type": "ap",
            "position": {"x": 0.0, "y": 0.0, "z": 1.5},
            "channel": 42,
            "bandwidth": 80,
            "applications": [],
            "agents": []
        }
        nodes = [ap_node]

        # generate stas
        for i in range(num_stas):
            sta_node = {
                "id": i + 1,
                "type": "sta",
                "position": {"x": x_list[i], "y": y_list[i], "z": 1.5},
                "channel": 42,
                "bandwidth": 80,
                "applications": [{"name": "ping", "dst": 0, "qos": "VO", "start_time": 0.5, "end_time": 0.8},
                                 {"name": f"saturated{int(num_stas/2)*2}", 
                                  "dst": 0, "qos": "BE", "start_time": 1.0, "end_time": 100.0}],
                "agents": []
            }
            if i < num_panlink_stas:
                sta_node["type"] = "panlink_sta"
                sta_node["agents"].append({
                    "type": "channel_access",
                    "period": 0.0,
                    "enable_learning": True,
                    "pretrained_model": ["RL_CHANNEL_ACCESS_actor.pt", "RL_CHANNEL_ACCESS_critic.pt"]
                })
            nodes.append(sta_node)

        with open(output_file, 'w') as f:
            json.dump({"nodes": nodes}, f, indent=4)
        return output_file


if __name__ == "__main__":
    config_file = "contrib/panlink/configs/panlink_channel_access_saturated.json"
    output_file = "contrib/panlink/configs/panlink_test.json"

    config_generator = ConfigGenerator(config_file)
    if config_generator.variable is not None:
        config_generator.generate(output_file, init=False)