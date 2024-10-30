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
PanlinkFrameExchangeManager::SetWifiPhy(const Ptr<WifiPhy> phy)
{
    NS_LOG_FUNCTION(this << phy);
    HeFrameExchangeManager::SetWifiPhy(phy);

    m_phy->SetReceiveOkCallback(MakeCallback(&PanlinkFrameExchangeManager::Sensing, this));
}

void
PanlinkFrameExchangeManager::Sensing(Ptr<const WifiPsdu> psdu,
                                     RxSignalInfo rxSignalInfo,
                                     const WifiTxVector& txVector,
                                     const std::vector<bool>& perMpduStatus)
{
    NS_LOG_FUNCTION(
        this << psdu << rxSignalInfo << txVector << perMpduStatus.size()
             << std::all_of(perMpduStatus.begin(), perMpduStatus.end(), [](bool v) { return v; }));
}

} // namespace ns3
