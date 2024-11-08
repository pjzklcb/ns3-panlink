/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-sta-wifi-mac.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkStaWifiMac");

NS_OBJECT_ENSURE_REGISTERED(PanlinkStaWifiMac);

TypeId
PanlinkStaWifiMac::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::PanlinkStaWifiMac")
            .SetParent<StaWifiMac>()
            .SetGroupName("Panlink")
            .AddConstructor<PanlinkStaWifiMac>();
    return tid;
}

PanlinkStaWifiMac::PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

PanlinkStaWifiMac::~PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
