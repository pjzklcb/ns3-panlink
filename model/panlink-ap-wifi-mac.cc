/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-ap-wifi-mac.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkApWifiMac");

NS_OBJECT_ENSURE_REGISTERED(PanlinkApWifiMac);

TypeId
PanlinkApWifiMac::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::PanlinkApWifiMac")
            .SetParent<ApWifiMac>()
            .SetGroupName("Panlink")
            .AddConstructor<PanlinkApWifiMac>();
    return tid;
}

PanlinkApWifiMac::PanlinkApWifiMac()
{
    NS_LOG_FUNCTION(this);
}

PanlinkApWifiMac::~PanlinkApWifiMac()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
