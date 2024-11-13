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
    static TypeId tid =
        TypeId("ns3::PanlinkTxop")
            .SetParent<QosTxop>()
            .SetGroupName("Panlink")
            .AddConstructor<PanlinkTxop>()
            .AddTraceSource(
                "WaitedSlots",
                "Trace source for values of cw, backoff, waited slots and deferred times",
                MakeTraceSourceAccessor(&PanlinkTxop::m_waitedSlotsTrace),
                "ns3::PanlinkTxop::WaitedSlotsValueTracedCallback");
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
PanlinkTxop::GetEnablePanlinkChannelAccess() const
{
    auto staMac = DynamicCast<PanlinkStaWifiMac>(m_mac);
    if (staMac && staMac->GetEnablePanlinkChannelAccess())
    {
        return true;
    }

    return false;
}

void
PanlinkTxop::GenerateBackoff(uint8_t linkId)
{
    if (!GetEnablePanlinkChannelAccess())
    {
        QosTxop::GenerateBackoff(linkId);
        return;
    }
    
    auto& link = GetLink(linkId);
    NS_LOG_FUNCTION(this << linkId << link.nDeferredAccess << link.waitedSlots);

    // trace last access's total number of waited slots as equivalent backoff
    m_backoffTrace(link.waitedSlots, linkId);

    // if (GetEnableChannelAccess())
    // {
    //     GenerateNewBackoff(linkId);
    // }
    uint32_t backoff = m_rng->GetInteger(0, GetCw(linkId));

    // reset access-related values
    StartBackoffNow(backoff, linkId);
    link.nDeferredAccess = 0;
    link.waitedSlots = 0;
    m_waitedSlotsTrace(link.cw, link.nDeferredAccess, link.waitedSlots, backoff, linkId);
}

void
PanlinkTxop::UpdateBackoffSlotsNow(uint32_t nSlots, Time backoffUpdateBound, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << nSlots << backoffUpdateBound << linkId);
    auto& link = GetLink(linkId);

    link.backoffSlots -= nSlots;
    link.backoffStart = backoffUpdateBound;
    link.waitedSlots += nSlots;
    NS_LOG_DEBUG("update slots=" << nSlots << ", total waited " << link.waitedSlots
                                 << " slots, backoff=" << link.backoffSlots);
}

void
PanlinkTxop::GenerateNewBackoff(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    if (GetEnablePanlinkChannelAccess())
    {
        auto& link = GetLink(linkId);

        link.nDeferredAccess += 1;
        link.backoffSlots = m_rng->GetInteger(0, link.cw - link.waitedSlots);
        NS_LOG_DEBUG("this access has deferred "
                     << link.nDeferredAccess << " times, waited " << link.waitedSlots
                     << " slots, and new backoff=" << link.backoffSlots);
        m_waitedSlotsTrace(link.cw, link.nDeferredAccess, link.waitedSlots, link.backoffSlots, linkId);
    }
}

} // namespace ns3
