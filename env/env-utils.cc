/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */


#include "env-utils.h"

#include "ns3/node.h"
#include "ns3/node-list.h"
#include "ns3/wifi-net-device.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvUtils");

uint32_t
ContextToNodeId(std::string context)
{
    std::string sub = context.substr(10);
    uint32_t pos = sub.find("/Device");
    return std::stoi(sub.substr(0, pos));
}

Mac48Address
ContextToMac(std::string context)
{
    uint32_t nodeId = ContextToNodeId(context);
    Ptr<Node> node = NodeList::GetNode(nodeId);
    Ptr<WifiNetDevice> device;
    for (uint32_t i = 0; i < node->GetNDevices(); i++)
    {
        device = DynamicCast<WifiNetDevice>(node->GetDevice(i));
        if (device)
        {
            break;
        }
    }
    return Mac48Address::ConvertFrom(device->GetAddress());
}

} // namespace ns3