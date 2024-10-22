/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-utils.h"

#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/wifi-mac.h"
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

uint32_t
MacAddressToNodeId(std::map<Mac48Address, uint32_t>& map, Mac48Address address)
{
    auto it = map.find(address);
    if (it == map.end())
    {
        for (uint32_t i = 0; i < NodeList::GetNNodes(); i++)
        {
            Ptr<Node> node = NodeList::GetNode(i);
            for (uint32_t j = 0; j < node->GetNDevices(); j++)
            {
                auto device = DynamicCast<WifiNetDevice>(node->GetDevice(j));
                if (device && device->GetMac()->GetAddress() == address)
                {
                    map[address] = i;
                    return i;
                }
            }
        }

        NS_ABORT_MSG("Mac address " << address << " does not exist.");
        return 0;
    }
    else
    {
        return it->second;
    }
}

AcIndex
StringToAcIndex(std::string ac)
{
    if (ac == "BE")
        return AC_BE;
    else if (ac == "BK")
        return AC_BK;
    else if (ac == "VI")
        return AC_VI;
    else if (ac == "VO")
        return AC_VO;
    else
        return AC_BE_NQOS;
}

std::string
AcIndexToString(AcIndex ac)
{
    switch (ac)
    {
    case AC_BE:
        return "BE";
    case AC_BK:
        return "BK";
    case AC_VI:
        return "VI";
    case AC_VO:
        return "VO";
    default:
        return "NON_QOS";
    }
}

AcIndex 
ContextToAcIndex(std::string context)
{
    uint32_t pos = context.find("_Txop");
    std::string ac = context.substr(pos-2, 2);
    return StringToAcIndex(ac);
}

} // namespace ns3