/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-sta-wifi-mac.h"

#include "panlink-frame-exchange-manager.h"

#include "ns3/eht-frame-exchange-manager.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkStaWifiMac");

NS_OBJECT_ENSURE_REGISTERED(PanlinkStaWifiMac);

TypeId
PanlinkStaWifiMac::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkStaWifiMac")
                            .SetParent<StaWifiMac>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkStaWifiMac>();
    return tid;
}

PanlinkStaWifiMac::PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);

    m_sensingMiddle = Create<MacSensingMiddle>();
}

PanlinkStaWifiMac::~PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

void
PanlinkStaWifiMac::DoDispose()
{
    NS_LOG_FUNCTION(this);
    m_sensingMiddle = nullptr;

    StaWifiMac::DoDispose();
}

void
PanlinkStaWifiMac::DoCompleteConfig()
{
    NS_LOG_FUNCTION(this);

    for (auto& [id, link] : GetLinks())
    {
        auto panlinkFeManager = DynamicCast<PanlinkFrameExchangeManager>(link->feManager);
        panlinkFeManager->SetMacSensingMiddle(m_sensingMiddle);
    }
}

} // namespace ns3
