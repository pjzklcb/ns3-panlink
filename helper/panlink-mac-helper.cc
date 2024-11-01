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

    std::vector<Ptr<SensingMac>> sensingMacs;
    auto sensingMac = m_sensingMac;
    for (uint8_t linkId = 0; linkId < mac->GetNLinks(); ++linkId)
    {
        auto sem = sensingMac.Create<SensingMac>();
        sensingMacs.emplace_back(sem);

        auto fem = mac->GetFrameExchangeManager(linkId);
        if (auto panlinkFem = DynamicCast<PanlinkFrameExchangeManager>(fem))
        {
            panlinkFem->SetSensingMac(sem);
        }
    }

    // if (auto apMac = DynamicCast<PanlinkApWifiMac>(mac))
    // {
    //   apMac->SetSensingMacs(sensingMacs);
    // }
    if (auto staMac = DynamicCast<PanlinkStaWifiMac>(mac))
    {
        staMac->SetSensingMacs(sensingMacs);
    }

    return mac;
}

} // namespace ns3
