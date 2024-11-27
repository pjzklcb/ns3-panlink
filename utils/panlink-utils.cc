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

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkUtils");

std::string
SnakeToCamel(const std::string& str)
{
    std::string result;
    bool capitalizeNext = false;

    for (char c : str)
    {
        if (c == '_')
        {
            capitalizeNext = true;
        }
        else
        {
            if (capitalizeNext)
            {
                result += std::toupper(c);
                capitalizeNext = false;
            }
            else
            {
                result += c;
            }
        }
    }

    if (!result.empty())
    {
        result[0] = std::toupper(result[0]);
    }

    return result;
}

rapidjson::Document
ReadPanlinkFromJson(std::string filename)
{
    rapidjson::Document document;

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        NS_ABORT_MSG("Failed to open " << filename);
    }

    // Read the JSON file into a string
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parse the JSON string
    document.Parse(json_str.c_str());

    if (document.HasParseError())
    {
        NS_ABORT_MSG("Failed to parse " << filename << ": "
                                        << rapidjson::GetParseErrorFunc(document.GetParseError()));
    }

    if (!document.IsObject() || !document.HasMember("nodes") || !document["nodes"].IsArray())
    {
        NS_ABORT_MSG("Failed to get nodes from " << filename);
    }

    return document;
}

std::string
ParseRandomVariableStream(const rapidjson::Value& item)
{
    std::string rand;
    if (item.IsDouble())
    {
        rand = "ns3::ConstantRandomVariable[Constant=" + std::to_string(item.GetDouble()) + "]";
        return rand;
    }
    if (item.IsInt())
    {
        rand = "ns3::ConstantRandomVariable[Constant=" + std::to_string(item.GetInt()) + "]";
        return rand;
    }

    if (!item.IsObject())
    {
        NS_ABORT_MSG("Missing distribution member in json");
        return rand;
    }

    std::string dist = SnakeToCamel(item["dist"].GetString());
    // check if distribution exists
    std::stringstream ss;
    ss << "ns3::" << dist << "RandomVariable";
    try
    {
        TypeId tid = TypeId::LookupByName(ss.str());
        NS_LOG_DEBUG(dist << " has typeId=" << tid);
    }
    catch (const std::exception& e)
    {
        NS_ABORT_MSG("Unknown distribution in json");
    }

    // create the random variable stream
    ss << "[";
    for (auto it = item.MemberBegin(); it != item.MemberEnd(); ++it)
    {
        // skip the "dist" memeber
        if (std::string(it->name.GetString()) == "dist")
        {
            continue;
        }
        std::string key = SnakeToCamel(it->name.GetString());
        std::string value = std::to_string(it->value.GetDouble());
        ss << key << "=" << value << "|";
    }

    ss.seekp(-1, std::ios_base::end);
    ss << "]";
    rand = ss.str();

    return rand;
}

SrcApp
ParseApplicationFromJson(const rapidjson::Value& item, uint32_t dstNodeId, uint16_t port)
{
    SrcApp app;

    if (!item.IsObject())
    {
        NS_ABORT_MSG("Application in json is not an object!");
        return app;
    }

    if (!item.HasMember("name") || !item.HasMember("qos") || !item.HasMember("start_time") ||
        !item.HasMember("end_time"))
    {
        NS_ABORT_MSG("Missing application member in json");
        return app;
    }

    uint64_t maxPackets = 0;
    if (item.HasMember("max_packets") && item["max_packets"].GetUint() > 0)
    {
        maxPackets = item["max_packets"].GetUint();
    }

    app = {item["name"].GetString(),
           dstNodeId,
           port, // port number
           StringToAcIndex(item["qos"].GetString()),
           item["start_time"].GetDouble(),
           item["end_time"].GetDouble(),
           maxPackets};

    return app;
}

uint16_t
AcIndexToTos(AcIndex ac)
{
    switch (ac)
    {
    case AC_BE:
        return 0x70;
    case AC_BK:
        return 0x28;
    case AC_VI:
        return 0xb8;
    case AC_VO:
        return 0xc0;
    default:
        return 0;
    }
}

// ============================================
// TopologyRecorder
// ============================================
TopologyRecorder::TopologyRecorder()
{
}

TopologyRecorder::~TopologyRecorder()
{
}

void
TopologyRecorder::Add(uint32_t bssid, uint32_t nodeId, std::string type, double x, double y)
{
    NodeInfo nodeInfo = {bssid, nodeId, type, x, y};
    m_nodes.push_back(nodeInfo);
}

void
TopologyRecorder::WriteToJson(std::string filename)
{
    rapidjson::Document doc;
    doc.SetObject();

    rapidjson::Value nodes;
    nodes.SetArray();

    auto& allocator = doc.GetAllocator();
    for (auto& node : m_nodes)
    {
        rapidjson::Value nodeInfo;
        nodeInfo.SetObject();
        nodeInfo.AddMember("bssid", node.m_bssid, allocator);
        nodeInfo.AddMember("id", node.m_nodeId, allocator);
        nodeInfo.AddMember("type", rapidjson::Value(node.m_type.c_str(), allocator), allocator);
        nodeInfo.AddMember("loc_x", node.m_x, allocator);
        nodeInfo.AddMember("loc_y", node.m_y, allocator);
        nodes.PushBack(nodeInfo, allocator);
    }

    doc.AddMember("nodes", nodes, allocator);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);

    std::ofstream file(filename);
    file << buffer.GetString();
}

} // namespace ns3
