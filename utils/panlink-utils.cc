/*
 * Copyright (c) 2024
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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

std::vector<PanlinkNode>
ReadNodesFromJson(std::string filename)
{
    std::vector<PanlinkNode> nodes;

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
            PanlinkNode panlinkNode;
            panlinkNode.m_id = node["id"].GetUint();
            panlinkNode.m_type = node["type"].GetString();
            panlinkNode.m_xPosition = node["position"]["x"].GetDouble();
            panlinkNode.m_yPosition = node["position"]["y"].GetDouble();
            panlinkNode.m_zPosition = node["position"]["z"].GetDouble();
            panlinkNode.m_channel = node["channel"].GetUint();
            panlinkNode.m_bandwidth = node["bandwidth"].GetUint();
            panlinkNode.m_applications = std::vector<SrcApp>();
            panlinkNode.m_agents = std::vector<AgentType>();

            // Iterate over each application
            rapidjson::Value applications = node["applications"].GetArray();
            for (rapidjson::SizeType i = 0; i < applications.Size(); i++)
            {
                auto& app = applications[i];
                SrcApp srcApp = {app["name"].GetString(),
                                 app["dst"].GetUint(),
                                 50000 + uint(i), // port number
                                 app["start_time"].GetDouble(),
                                 app["end_time"].GetDouble()};

                panlinkNode.m_applications.push_back(srcApp);
            }
            // Iterate over each agent
            for (auto& agent : node["agents"].GetArray())
            {
                std::string agentType = agent["type"].GetString();
                if (agentType == "CCA" || agentType == "cca_sensitivity" || 
                    agentType == "CCA-Sensitivity")
                {
                    panlinkNode.m_agents.push_back(RL_CCA_SENSITIVITY);
                }
                else if (agentType == "channel_access" || agentType == "Channel-Access")
                {
                    panlinkNode.m_agents.push_back(RL_CHANNEL_ACCESS);
                }
                else
                {
                    NS_LOG_ERROR("Unknown agent type: " << agent.GetString());
                    return nodes;
                }
                // TODO: Add other agents
            }
            std::cout << "Node ID: " << panlinkNode.m_id << ", Node type: " << panlinkNode.m_type
                      << ", Position: " << panlinkNode.m_xPosition << ", "
                      << panlinkNode.m_yPosition << ", " << panlinkNode.m_zPosition
                      << ", Channel: " << panlinkNode.m_channel
                      << ", Bandwidth: " << panlinkNode.m_bandwidth << std::endl;
            nodes.push_back(panlinkNode);
        }
    }

    return nodes;
}

BssMap
ReadBssMapFromJson(std::string filename)
{
    BssMap bssMap;

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        NS_LOG_ERROR("Failed to open " << filename);
        return bssMap;
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
        return bssMap;
    }

    if (document.IsObject() && document.HasMember("bss") && document["bss"].IsArray())
    {
        // Iterate over each bss
        for (auto& bss : document["bss"].GetArray())
        {
            BssInfo bssInfo;

            // Extract the bss's properties from the JSON object
            bssInfo.m_bssIdx = bss["bss_idx"].GetInt();
            bssInfo.m_ssid = bss["ssid"].GetString();
            bssInfo.m_bssIp = bss["bss_ip"].GetString();
            bssInfo.m_nodes = std::vector<PanlinkNode>();

            std::cout << "BSS ID: " << bssInfo.m_bssIdx << ", SSID: " << bssInfo.m_ssid
                      << ", BSS IP: " << bssInfo.m_bssIp << std::endl;

            // Iterate over each node
            for (auto& node : bss["nodes"].GetArray())
            {
                PanlinkNode panlinkNode;
                panlinkNode.m_id = node["node_idx"].GetInt();
                panlinkNode.m_bssIdx = bssInfo.m_bssIdx;
                panlinkNode.m_type = node["node_type"].GetString();
                panlinkNode.m_xPosition = node["position"]["x"].GetDouble();
                panlinkNode.m_yPosition = node["position"]["y"].GetDouble();
                panlinkNode.m_zPosition = node["position"]["z"].GetDouble();
                panlinkNode.m_channel = node["channel"].GetInt();
                panlinkNode.m_bandwidth = node["bandwidth"].GetInt();
                panlinkNode.m_applications = std::vector<SrcApp>();
                panlinkNode.m_agents = std::vector<AgentType>();

                // Iterate over each application
                for (auto& app : node["applications"].GetArray())
                {
                    SrcApp srcApp;
                    srcApp.m_name = app["name"].GetString();
                    srcApp.m_dst = app["dst"].GetInt();
                    srcApp.m_port = app["port"].GetInt();
                    srcApp.m_startTime = app["start_time"].GetDouble();
                    srcApp.m_endTime = app["end_time"].GetDouble();
                    panlinkNode.m_applications.push_back(srcApp);
                }
                // Iterate over each agent
                for (auto& agent : node["agents"].GetArray())
                {
                    std::string agentType = agent.GetString();
                    if (agentType == "CCA" || agentType == "cca_sensitivity" || 
                        agentType == "CCA-Sensitivity")
                    {
                        panlinkNode.m_agents.push_back(RL_CCA_SENSITIVITY);
                    }
                    else if (agentType == "channel_access" || agentType == "Channel-Access")
                    {
                        panlinkNode.m_agents.push_back(RL_CHANNEL_ACCESS);
                    }
                    else
                    {
                        NS_LOG_ERROR("Unknown agent type: " << agent.GetString());
                    }
                    // TODO: Add other agents
                }

                bssInfo.m_nodes.push_back(panlinkNode);

                std::cout << "Node ID: " << panlinkNode.m_id
                          << ", Node type: " << panlinkNode.m_type
                          << ", Position: " << panlinkNode.m_xPosition << ", "
                          << panlinkNode.m_yPosition << ", " << panlinkNode.m_zPosition
                          << ", Channel: " << panlinkNode.m_channel
                          << ", Bandwidth: " << panlinkNode.m_bandwidth << std::endl;
            }

            // Add the BssInfo to the BssMap
            bssMap[bssInfo.m_bssIdx] = bssInfo;
        }
    }

    return bssMap;
}

AppMap
ReadAppInfoFromJson(std::string filename)
{
    AppMap appMap;

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        NS_LOG_ERROR("Failed to open " << filename);
        return appMap;
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
        return appMap;
    }

    if (document.IsObject() && document.HasMember("application") &&
        document["application"].IsObject())
    {
        for (auto& app : document["application"].GetArray())
        {
            std::string appName = app["name"].GetString();

            auto it = appMap.find(appName);
            if (it != appMap.end())
            {
                NS_LOG_ERROR("Duplicated application name: " << appName << " in " << filename);
                return appMap;
            }

            AppInfo appInfo;
            appInfo.m_type = app["type"].GetString();
            appInfo.m_params = std::unordered_map<std::string, double>();

            if (app.HasMember("params") && app["params"].IsObject())
            {
                for (auto& param : app["params"].GetArray())
                {
                    // udp params: {max_packets, interval, packet_size}
                    if (appInfo.m_type == "udp")
                    {
                        if (param.HasMember("max_packets") && param["max_packets"].IsInt())
                        {
                            appInfo.m_params["MaxPackets"] = param["max_packets"].GetInt();
                        }
                        else
                        {
                            appInfo.m_params["MaxPackets"] = 0;
                        }

                        if (param.HasMember("interval") && param["interval"].IsDouble())
                        {
                            appInfo.m_params["Interval"] =
                                param["interval"].GetDouble(); // in seconds
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'interval' in " << appName << " in " << filename);
                        }

                        if (param.HasMember("packet_size") && param["packet_size"].IsInt())
                        {
                            appInfo.m_params["PacketSize"] =
                                param["packet_size"].GetInt(); // in bytes
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'packet_size' in " << appName << " in "
                                                                     << filename);
                        }
                    }
                    // constant params: {interval, packet_size, fragment_size}
                    else if (appInfo.m_type == "constant")
                    {
                        if (param.HasMember("interval") && param["interval"].IsDouble())
                        {
                            appInfo.m_params["Interval"] =
                                param["interval"].GetDouble(); // in seconds
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'interval' in " << appName << " in " << filename);
                        }

                        if (param.HasMember("burst_size") && param["burst_size"].IsInt())
                        {
                            appInfo.m_params["BurstSize"] =
                                param["burst_size"].GetInt(); // in bytes
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'burst_size' in " << appName << " in "
                                                                    << filename);
                        }

                        if (param.HasMember("fragment_size") && param["fragment_size"].IsInt())
                        {
                            appInfo.m_params["FragmentSize"] =
                                param["fragment_size"].GetInt(); // in bytes
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'fragment_size' in " << appName << " in "
                                                                       << filename);
                        }
                    }
                    // poisson params: {lambda}
                    else if (appInfo.m_type == "poisson")
                    {
                        if (param.HasMember("lambda") && param["lambda"].IsDouble())
                        {
                            appInfo.m_params["Lambda"] = param["lambda"].GetDouble();
                        }
                        else
                        {
                            NS_LOG_ERROR("Missing 'lambda' in " << appName << " in " << filename);
                        }
                    }
                    else
                    {
                        NS_LOG_ERROR("Unknown application type: " << appInfo.m_type);
                        return appMap;
                    }
                    // TODO: Add other application types
                }
            }
            else
            {
                NS_LOG_ERROR("Failed to find parameters for " << appInfo.m_type << " in "
                                                              << filename);
                return appMap;
            }

            appMap[appName] = appInfo;
        }
    }

    return appMap;
}

// uint32_t
// MacAddressToNodeId(Mac48Address address)
// {
//     for (uint32_t i = 0; i < apDevices.GetN(); i++)
//     {
//         if (apDevices.Get(i)->GetAddress() == address)
//         {
//             return apNodes.Get(i)->GetId();
//         }
//     }

//     auto it = m_staMacAddressToNodeId.find(address);
//     if (it != m_staMacAddressToNodeId.end())
//     {
//         return it->second;
//     }

//     NS_ABORT_MSG("Found no node having MAC address " << address);
// }

// /**
//  * Parse context strings of the form "/NodeList/x/DeviceList/x/..." to extract the NodeId
//  * integer
//  *
//  * \param context The context to parse.
//  * \return the NodeId
//  */
// uint32_t
// ContextToNodeId(std::string context)
// {
//     std::string sub = context.substr(10);
//     uint32_t pos = sub.find("/Device");
//     return std::stoi(sub.substr(0, pos));
// }

// /**
//  * Parse context strings of the form "/NodeList/x/DeviceList/x/..." and fetch the Mac address
//  *
//  * \param context The context to parse.
//  * \return the device MAC address
//  */
// Mac48Address
// ContextToMac(std::string context)
// {
//     std::string sub = context.substr(10);
//     uint32_t pos = sub.find("/Device");
//     uint32_t nodeId = std::stoi(sub.substr(0, pos));
//     Ptr<Node> n = NodeList::GetNode(nodeId);
//     Ptr<WifiNetDevice> d;
//     for (uint32_t i = 0; i < n->GetNDevices(); i++)
//     {
//         d = n->GetDevice(i)->GetObject<WifiNetDevice>();
//         if (d)
//         {
//             break;
//         }
//     }
//     return Mac48Address::ConvertFrom(d->GetAddress());
// }

// std::string
// AddressToString(const Address& addr)
// {
//     std::stringstream addressStr;
//     addressStr << InetSocketAddress::ConvertFrom(addr).GetIpv4() << ":"
//                << InetSocketAddress::ConvertFrom(addr).GetPort();
//     return addressStr.str();
// }

} // namespace ns3
