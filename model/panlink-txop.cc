/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-txop.h"

#include "panlink-sta-wifi-mac.h"

#include "ns3/channel-access-env-agent.h"
#include "ns3/env-agent-manager.h"

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
                "WaitedSlotsTrace",
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

void
PanlinkTxop::SetBackoffSlots(uint32_t backoff, uint8_t linkId)
{
    GetLink(linkId).backoffSlots = backoff;
}

uint32_t
PanlinkTxop::GetBackoffSlots(uint8_t linkId) const
{
    return GetLink(linkId).backoffSlots;
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

    // reset access-related values
    link.backoffStart = Simulator::Now();
    link.nDeferredAccess = 0;
    link.waitedSlots = 0;
    // link.backoffSlots = m_rng->GetInteger(0, GetCw(linkId));

    auto obs = ChannelAccessEnvAgent::ChannelAccessObs();
    obs.linkId = linkId;
    obs.ac = GetAccessCategory();
    obs.cw = link.cw;
    obs.nDeferredAccess = 0;
    obs.waitedSlots = 0;

    uint32_t nodeId = m_mac->GetDevice()->GetNode()->GetId();
    EnvAgentManager::Get()->EnvInfoExchange(nodeId, RL_CHANNEL_ACCESS, obs);

    m_waitedSlotsTrace(link.nDeferredAccess, link.waitedSlots, link.backoffSlots, linkId);
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
        // link.backoffSlots = m_rng->GetInteger(0, link.cw - link.waitedSlots);
        
        auto obs = ChannelAccessEnvAgent::ChannelAccessObs();
        obs.linkId = linkId;
        obs.ac = GetAccessCategory();
        obs.cw = link.cw;
        obs.nDeferredAccess = link.nDeferredAccess;
        obs.waitedSlots = link.waitedSlots;

        uint32_t nodeId = m_mac->GetDevice()->GetNode()->GetId();
        EnvAgentManager::Get()->EnvInfoExchange(nodeId, RL_CHANNEL_ACCESS, obs);

        NS_LOG_DEBUG("this access has deferred "
                     << link.nDeferredAccess << " times, waited " << link.waitedSlots
                     << " slots, and new backoff=" << link.backoffSlots);
        m_waitedSlotsTrace(link.nDeferredAccess, link.waitedSlots, link.backoffSlots, linkId);
    }
}

} // namespace ns3
