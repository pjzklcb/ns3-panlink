/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_UTILS_H
#define PANLINK_UTILS_H

#include "ns3/wifi-module.h"
#include "ns3/env-utils.h"

namespace ns3
{

/**
 * The information of an application installed on a source node.
 */
struct SrcApp
{
    std::string m_name;
    uint32_t m_dst;
    uint32_t m_port;
    AcIndex m_ac;
    double m_startTime;
    double m_endTime;
};

typedef std::map<uint32_t /* src node id */, std::vector<SrcApp>> SrcAppMap;

/**
 * The information of an agent installed on a node.
 */
struct AgentInfo
{
    AgentType m_type;
    Time m_period{Seconds(0)};
};

/**
 * The information of a node specified in the configuration file.
 */
struct PanlinkNodeInfo
{
    uint32_t m_id;
    uint32_t m_bssIdx;
    std::string m_type; // PO, PC, AP or STA
    double m_xPosition;
    double m_yPosition;
    double m_zPosition;
    uint32_t m_channel;
    uint32_t m_bandwidth;
    std::vector<SrcApp> m_applications; // application type
    std::vector<AgentInfo> m_agents;
};

/**
 * \param filename the name of the JSON file
 * \return Panlink nodes configured in the JSON
 */
std::vector<PanlinkNodeInfo> ReadNodesFromJson(std::string filename);

} // namespace ns3

#endif /* PANLINK_UTILS_H */
