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
    static TypeId tid =
        TypeId("ns3::PanlinkStaWifiMac")
            .SetParent<StaWifiMac>()
            .SetGroupName("Panlink")
            .AddConstructor<PanlinkStaWifiMac>()
            .AddAttribute("EnablePanlinkChannelAccess",
                          "Enable the Panlink channel access strategy.",
                          TypeId::ATTR_GET |
                              TypeId::ATTR_CONSTRUCT, // prevent setting after construction
                          BooleanValue(false),
                          MakeBooleanAccessor(&PanlinkStaWifiMac::SetEnablePanlinkChannelAccess,
                                              &PanlinkStaWifiMac::GetEnablePanlinkChannelAccess),
                          MakeBooleanChecker());
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

void
PanlinkStaWifiMac::DoInitialize()
{
    NS_LOG_FUNCTION(this);

    if (m_sensingMac)
    {
        m_sensingMac->Initialize();
    }

    StaWifiMac::DoInitialize();
}

void
PanlinkStaWifiMac::DoDispose()
{
    NS_LOG_FUNCTION(this);

    if (m_sensingMac)
    {
        m_sensingMac->Dispose();
    }
    m_sensingMac = nullptr;

    StaWifiMac::DoDispose();
}

void
PanlinkStaWifiMac::SetSensingMac(const Ptr<SensingMac> sensingMac)
{
    NS_LOG_FUNCTION(this);

    m_sensingMac = sensingMac;
    // connect callbacks
    TraceConnectWithoutContext("AckedMpdu",
                               MakeCallback(&SensingMac::NotifyAckedMpdu, m_sensingMac));
    TraceConnectWithoutContext("DroppedMpdu",
                               MakeCallback(&SensingMac::NotifyDroppedMpdu, m_sensingMac));
    // TraceConnectWithoutContext("MpduResponseTimeout",
    //                            MakeCallback(&SensingMac::NotifyMpduTimeout, m_sensingMac));
    // TraceConnectWithoutContext("PsduResponseTimeout",
    //                            MakeCallback(&SensingMac::NotifyPsduTimeout, m_sensingMac));
    for (const auto& [aci, _] : wifiAcList)
    {
        GetTxopQueue(aci)->TraceConnect("Enqueue",
                                        AcIndexToString(aci),
                                        MakeCallback(&SensingMac::NotifyEdcaEnqueue, m_sensingMac));
        GetTxopQueue(aci)->TraceConnect("Expired",
                                        AcIndexToString(aci),
                                        MakeCallback(&SensingMac::NotifyEdcaExpired, m_sensingMac));
    }
    for (const auto& [id, link] : GetLinks())
    {
        link->phy->TraceConnect("PhyTxBegin",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyTxBegin, m_sensingMac));
        link->phy->TraceConnect("PhyTxEnd",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyTxEnd, m_sensingMac));
        link->phy->TraceConnect("PhyRxBegin",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyRxBegin, m_sensingMac));
        link->phy->TraceConnect("PhyRxEnd",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyRxEnd, m_sensingMac));
        link->phy->TraceConnect("PhyRxPpduDrop",
                                std::to_string(id),
                                MakeCallback(&SensingMac::NotifyPhyRxPpduDrop, m_sensingMac));
        link->phy->GetState()->TraceConnect(
            "State",
            std::to_string(id),
            MakeCallback(&SensingMac::NotifyPhyState, m_sensingMac));
    }
}

Ptr<SensingMac>
PanlinkStaWifiMac::GetSensingMac() const
{
    return m_sensingMac;
}

void
PanlinkStaWifiMac::SetEnablePanlinkChannelAccess(bool enable)
{
    m_panlinkChannelAccess = enable;
}

bool
PanlinkStaWifiMac::GetEnablePanlinkChannelAccess() const
{
    return m_panlinkChannelAccess;
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

} // namespace ns3
