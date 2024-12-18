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
PanlinkFrameExchangeManager::SetSensingMac(const Ptr<SensingMac> sensingMac)
{
    NS_LOG_FUNCTION(this << sensingMac);
    m_sensingMac = sensingMac;
}

void
PanlinkFrameExchangeManager::Receive(Ptr<const WifiPsdu> psdu,
                                     RxSignalInfo rxSignalInfo,
                                     const WifiTxVector& txVector,
                                     const std::vector<bool>& perMpduStatus)
{
    NS_LOG_FUNCTION(this << psdu << rxSignalInfo << txVector);

    FrameExchangeManager::Receive(psdu, rxSignalInfo, txVector, perMpduStatus);

    // Only process once after the whole PSDU is received.
    if (perMpduStatus.empty())
    {
        return;
    }

    // Capturing ACK or BA
    auto& hdr = psdu->GetHeader(0);
    if (hdr.IsAck() || hdr.IsBlockAck())
    {
        Mac48Address receiver = hdr.GetAddr1();
        m_sensingMac->NotifyLastTxOkSta(receiver, m_linkId);
    }

    if (!hdr.IsAck() && !hdr.IsCts() && !hdr.IsRts())
    {
        // Ack and CTS do not carry Addr2, the duration of CTS and RTS contains PPDU's length
        double rssi = rxSignalInfo.rssi;
        m_sensingMac->NotifyFrameInfo(*(psdu->begin()), hdr, rssi, m_linkId);
    }
}

} // namespace ns3
