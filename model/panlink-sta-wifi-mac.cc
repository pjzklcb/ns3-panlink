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
#include "ns3/env-utils.h"
#include "ns3/wifi-mac-queue.h"

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
PanlinkStaWifiMac::SetSensingMac(const Ptr<SensingMac> sensingMac)
{
    NS_LOG_FUNCTION(this);

    m_sensingMac = sensingMac;
    // connect callbacks
    TraceConnectWithoutContext("AckedMpdu", MakeCallback(&SensingMac::AckedMpdu, m_sensingMac));
    for (const auto& [ac, _] : wifiAcList)
    {
        GetTxopQueue(ac)->TraceConnect("Enqueue",
                                       AcIndexToString(ac),
                                       MakeCallback(&SensingMac::NotifyEdcaEnqueue, m_sensingMac));
    }
    for (const auto& [id, link] : GetLinks())
    {
        link->phy->TraceConnect("PhyTxBegin",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyTxBegin, m_sensingMac));
    }
}

Ptr<SensingMac>
PanlinkStaWifiMac::GetSensingMac() const
{
    return m_sensingMac;
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
    m_sensingMac->WifiMacSensing(mpdu, linkId);
}

} // namespace ns3
