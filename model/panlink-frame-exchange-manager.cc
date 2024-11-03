/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-frame-exchange-manager.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkFrameExchangeManager");

NS_OBJECT_ENSURE_REGISTERED(PanlinkFrameExchangeManager);

TypeId
PanlinkFrameExchangeManager::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkFrameExchangeManager")
                            .SetParent<HeFrameExchangeManager>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkFrameExchangeManager>();
    return tid;
}

PanlinkFrameExchangeManager::PanlinkFrameExchangeManager()
{
    NS_LOG_FUNCTION(this);
}

PanlinkFrameExchangeManager::~PanlinkFrameExchangeManager()
{
    NS_LOG_FUNCTION(this);
}

void
PanlinkFrameExchangeManager::DoDispose()
{
    NS_LOG_FUNCTION(this);
    m_sensingMac = nullptr;

    FrameExchangeManager::DoDispose();
}

void
PanlinkFrameExchangeManager::SetWifiPhy(const Ptr<WifiPhy> phy)
{
    NS_LOG_FUNCTION(this << phy);
    HeFrameExchangeManager::SetWifiPhy(phy);

    m_phy->SetReceiveOkCallback(MakeCallback(&PanlinkFrameExchangeManager::Sensing, this));
}

void 
PanlinkFrameExchangeManager::SetSensingMac(const Ptr<SensingMac> sensingMac)
{
    NS_LOG_FUNCTION(this << sensingMac);
    m_sensingMac = sensingMac;
}

void
PanlinkFrameExchangeManager::Sensing(Ptr<const WifiPsdu> psdu,
                                     RxSignalInfo rxSignalInfo,
                                     const WifiTxVector& txVector,
                                     const std::vector<bool>& perMpduStatus)
{
    NS_LOG_FUNCTION(this << psdu << rxSignalInfo << txVector << perMpduStatus.size());

    // Ignore ACK and CTS that does not contain TA, and RTS whose duration does not reflect the
    // channel occupancy of itself
    if (psdu->GetNMpdus() == 1)
    {
        auto& hdr = psdu->GetHeader(0);
        if (hdr.IsAck() || hdr.IsCts() || hdr.IsRts())
            return;
    }

    // Get frame's sender, RSSI and duration
    Mac48Address sender = psdu->GetAddr2();
    double rssi = rxSignalInfo.rssi;
    Time duration = psdu->GetDuration();

    m_sensingMac->FrameExchangeSensing(sender, rssi, duration, m_linkId);
}

} // namespace ns3
