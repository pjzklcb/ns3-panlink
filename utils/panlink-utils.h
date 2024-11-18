/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_UTILS_H
#define PANLINK_UTILS_H

#include "ns3/env-utils.h"
#include "ns3/wifi-module.h"

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

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
 * Converts a string from snake_case to CamelCase.
 *
 * @param str the string to convert
 */
std::string SnakeToCamel(const std::string& str);

/**
 * The information of an agent installed on a node.
 */
struct AgentInfo
{
    AgentType m_type;
    Time m_period{Seconds(0)};
};

typedef std::map<uint32_t /* src node id */, std::vector<AgentInfo>> AgentMap;

/**
 * @param filename the name of the JSON file
 * @return Panlink nodes info configured in the JSON
 */
rapidjson::Document ReadPanlinkFromJson(std::string filename);

/**
 * Get the Type of Service of the specified AC Index.
 * Check qos-utils.h
 *
 * @param ac the AC string {AC_BE:0x70, AC_BK:0x28, AC_VI:0xb8, AC_VO:0xc0}
 * @return the ToS
 */
uint16_t AcIndexToTos(AcIndex ac);

/**
 * Record the Topology of the simulation.
 */
class TopologyRecorder
{
  public:
    TopologyRecorder();
    ~TopologyRecorder();

    void Add(uint32_t bssid, uint32_t nodeId, std::string type, double x, double y);

    void WriteToJson(std::string filename);

  private:
    struct NodeInfo
    {
        uint32_t m_bssid;
        uint32_t m_nodeId;
        std::string m_type;
        double m_x;
        double m_y;
    };

    std::vector<NodeInfo> m_nodes;
};

} // namespace ns3

#endif /* PANLINK_UTILS_H */
