/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-mac-helper.h"

#include "ns3/panlink-frame-exchange-manager.h"
#include "ns3/panlink-sta-wifi-mac.h"
#include "ns3/sensing-mac.h"
#include "ns3/wifi-net-device.h"

namespace ns3
{

PanlinkMacHelper::PanlinkMacHelper()
{
    m_sensingMac.SetTypeId("ns3::SensingMac");
}

PanlinkMacHelper::~PanlinkMacHelper()
{
}

Ptr<WifiMac>
PanlinkMacHelper::Create(Ptr<WifiNetDevice> device, WifiStandard standard) const
{
    auto mac = WifiMacHelper::Create(device, standard);

    if (m_mac.GetTypeId() != PanlinkStaWifiMac::GetTypeId())
    // && m_mac.GetTypeId() != PanlinkApWifiMac::GetTypeId())
    {
        return mac;
    }

    auto sensingMac = m_sensingMac.Create<SensingMac>();
    sensingMac->SetWifiMac(mac);
    for (uint8_t linkId = 0; linkId < mac->GetNLinks(); ++linkId)
    {
        auto fem = mac->GetFrameExchangeManager(linkId);
        if (auto panlinkFem = DynamicCast<PanlinkFrameExchangeManager>(fem))
        {
            panlinkFem->SetSensingMac(sensingMac);
        }
    }

    std::cout << "checkpoint" << std::endl;
    // if (auto apMac = DynamicCast<PanlinkApWifiMac>(mac))
    // {
    //   apMac->SetSensingMac(sensingMac);
    // }
    if (auto staMac = DynamicCast<PanlinkStaWifiMac>(mac))
    {
        staMac->SetSensingMac(sensingMac);
    }

    return mac;
}

} // namespace ns3
