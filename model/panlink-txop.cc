/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-txop.h"

#include "panlink-sta-wifi-mac.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkTxop");

NS_OBJECT_ENSURE_REGISTERED(PanlinkTxop);

TypeId
PanlinkTxop::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkTxop")
                            .SetParent<QosTxop>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkTxop>();
    return tid;
}

PanlinkTxop::PanlinkTxop()
{
    NS_LOG_FUNCTION(this);
}

PanlinkTxop::~PanlinkTxop()
{
    NS_LOG_FUNCTION(this);
}

std::unique_ptr<Txop::LinkEntity>
PanlinkTxop::CreateLinkEntity() const
{
    return std::make_unique<PanLinkEntity>();
}

PanlinkTxop::PanLinkEntity&
PanlinkTxop::GetLink(uint8_t linkId) const
{
    return static_cast<PanLinkEntity&>(Txop::GetLink(linkId));
}

uint32_t
PanlinkTxop::GetWaitedSlots(uint8_t linkId) const
{
    return GetLink(linkId).waitedSlots;
}

uint32_t
PanlinkTxop::GetNDeferredAccess(uint8_t linkId) const
{
    return GetLink(linkId).nDeferredAccess;
}

bool
PanlinkTxop::GetEnableChannelAccess() const
{
    auto staMac = DynamicCast<PanlinkStaWifiMac>(m_mac);
    if (staMac && staMac->GetEnableChannelAccess())
    {
        return true;
    }

    return false;
}

void
PanlinkTxop::GenerateBackoff(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    // if (GetEnableChannelAccess())
    // {
    //     GenerateNewBackoff(linkId);
    //     return;
    // }

    Txop::GenerateBackoff(linkId);
}

void
PanlinkTxop::GenerateNewBackoff(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    if (GetEnableChannelAccess())
    {
        uint32_t waitedSlots = GetWaitedSlots(linkId);
        uint32_t backoff = m_rng->GetInteger(0, GetCw(linkId) - waitedSlots);
        NS_LOG_DEBUG("waitedSlots=" << waitedSlots << ", new backoff=" << backoff);
    }
}

} // namespace ns3
