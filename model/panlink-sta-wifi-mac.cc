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

    m_rxMiddle->SetForwardCallback(MakeCallback(&PanlinkStaWifiMac::Sensing, this));
}

PanlinkStaWifiMac::~PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

void
PanlinkStaWifiMac::SetSensingMacs(const std::vector<Ptr<SensingMac>>& sensingMacs)
{
    NS_LOG_FUNCTION(this);

    for (auto sensingIt = sensingMacs.cbegin(); auto& [id, link] : GetLinks())
    {
        auto sensingMac = GetPanStaLink(link).sensingMac;
        sensingMac = *sensingIt++;
        sensingMac->SetWifiMac(this);
        sensingMac->SetLinkId(id);
        // connect callbacks
        link->feManager->SetAckedMpduCallback(MakeCallback(&SensingMac::AckedMpdu, sensingMac));
    }
    auto sensingMac = GetLink(SINGLE_LINK_OP_ID).sensingMac;
    for (const auto& [ac, _] : wifiAcList)
    {
        GetQosTxop(ac)->GetBaManager()->SetTxOkCallback(
            MakeCallback(&SensingMac::AckedMpdu, sensingMac));
    }
}

Ptr<SensingMac>
PanlinkStaWifiMac::GetSensingMac(uint8_t linkId) const
{
    return GetLink(linkId).sensingMac;
}

PanlinkStaWifiMac::PanStaLinkEntity::~PanStaLinkEntity()
{
    NS_LOG_FUNCTION_NOARGS();
}

std::unique_ptr<WifiMac::LinkEntity>
PanlinkStaWifiMac::CreateLinkEntity() const
{
    return std::make_unique<PanStaLinkEntity>();
}

PanlinkStaWifiMac::PanStaLinkEntity&
PanlinkStaWifiMac::GetLink(uint8_t linkId) const
{
    return static_cast<PanStaLinkEntity&>(WifiMac::GetLink(linkId));
}

PanlinkStaWifiMac::PanStaLinkEntity&
PanlinkStaWifiMac::GetPanStaLink(const std::unique_ptr<WifiMac::LinkEntity>& link) const
{
    return static_cast<PanStaLinkEntity&>(*link);
}

void
PanlinkStaWifiMac::Sensing(Ptr<const WifiMpdu> mpdu, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << *mpdu << linkId);
    GetSensingMac(linkId)->WifiMacSensing(mpdu, linkId);
}

} // namespace ns3
