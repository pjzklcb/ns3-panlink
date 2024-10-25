/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-utils.h"

#include <fstream>
#include <iostream>
#include <rapidjson/document.h>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkUtils");

std::vector<PanlinkNodeInfo>
ReadNodesFromJson(std::string filename)
{
    std::vector<PanlinkNodeInfo> nodes;

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        NS_LOG_ERROR("Failed to open " << filename);
        return nodes;
    }

    // Read the JSON file into a string
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parse the JSON string
    rapidjson::Document document;
    document.Parse(json_str.c_str());

    if (document.HasParseError())
    {
        NS_LOG_ERROR("Failed to parse " << filename << ": "
                                        << rapidjson::GetParseErrorFunc(document.GetParseError()));
        return nodes;
    }

    if (document.IsObject() && document.HasMember("nodes") && document["nodes"].IsArray())
    {
        // Iterate over each node
        for (auto& node : document["nodes"].GetArray())
        {
            PanlinkNodeInfo nodeInfo;
            nodeInfo.m_id = node["id"].GetUint();
            nodeInfo.m_type = node["type"].GetString();
            nodeInfo.m_xPosition = node["position"]["x"].GetDouble();
            nodeInfo.m_yPosition = node["position"]["y"].GetDouble();
            nodeInfo.m_zPosition = node["position"]["z"].GetDouble();
            nodeInfo.m_channel = node["channel"].GetUint();
            nodeInfo.m_bandwidth = node["bandwidth"].GetUint();
            nodeInfo.m_applications = std::vector<SrcApp>();
            nodeInfo.m_agents = std::vector<AgentInfo>();

            // Iterate over each application
            rapidjson::Value applications = node["applications"].GetArray();
            for (rapidjson::SizeType i = 0; i < applications.Size(); i++)
            {
                auto& app = applications[i];
                SrcApp srcApp = {app["name"].GetString(),
                                 app["dst"].GetUint(),
                                 50000 + uint(i), // port number
                                 StringToAcIndex(app["qos"].GetString()),
                                 app["start_time"].GetDouble(),
                                 app["end_time"].GetDouble()};

                nodeInfo.m_applications.push_back(srcApp);
            }
            // Iterate over each agent
            for (auto& agent : node["agents"].GetArray())
            {
                std::string agentType = agent["type"].GetString();
                double period = agent["period"].GetDouble();
                AgentType type;
                if (agentType == "CCA" || 
                    agentType == "cca_sensitivity" || agentType == "cca-sensitivity" ||
                    agentType == "CCA-Sensitivity" || agentType == "CCA_Sensitivity")
                {
                    type = RL_CCA_SENSITIVITY;
                }
                else if (agentType == "channel_access" || agentType == "Channel-Access"||
                         agentType == "channel access" || agentType == "Channel Access")
                {
                    type = RL_CHANNEL_ACCESS;
                }
                else
                {
                    NS_LOG_ERROR("Unknown agent type: " << agent.GetString());
                    return nodes;
                }
                // TODO: Add other agents

                nodeInfo.m_agents.push_back({type, Seconds(period)});
            }
            std::cout << "Node ID: " << nodeInfo.m_id << ", Node type: " << nodeInfo.m_type
                      << ", Position: (" << nodeInfo.m_xPosition << ", "
                      << nodeInfo.m_yPosition << ", " << nodeInfo.m_zPosition
                      << "), Channel: " << nodeInfo.m_channel
                      << ", Bandwidth: " << nodeInfo.m_bandwidth << std::endl;
            nodes.push_back(nodeInfo);
        }
    }

    return nodes;
}

} // namespace ns3
