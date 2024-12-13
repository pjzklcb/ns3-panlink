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
PanlinkTxop::DoInitialize()
{
    NS_LOG_FUNCTION(this);

    if (!GetEnablePanlinkChannelAccess())
    {
        QosTxop::DoInitialize();
        return;
    }

    for (const auto& [id, link] : GetLinks())
    {
        ResetCw(id);
    }
}

void
PanlinkTxop::NotifyChannelReleased(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << +linkId);

    if (!GetEnablePanlinkChannelAccess())
    {
        QosTxop::NotifyChannelReleased(linkId);
        return;
    }

    auto& link = GetLink(linkId);

    if (link.startTxop)
    {
        NS_LOG_DEBUG("Terminating TXOP. Duration = " << Simulator::Now() - *link.startTxop);
        // m_txopTrace(*link.startTxop, Simulator::Now() - *link.startTxop, linkId);
    }

    // generate a new backoff value if either the TXOP duration is not null (i.e., some frames
    // were transmitted) or no frame was transmitted but the queue actually contains frame to
    // transmit and the user indicated that a backoff value should be generated in this situation.
    // This behavior reflects the following specs text (Sec. 35.3.16.4 of 802.11be D4.0):
    // An AP or non-AP STA affiliated with an MLD that has gained the right to initiate the
    // transmission of a frame as described in 10.23.2.4 (Obtaining an EDCA TXOP) for an AC but
    // does not transmit any frame corresponding to that AC for the reasons stated above may:
    // - invoke a backoff for the EDCAF associated with that AC as allowed per h) of 10.23.2.2
    //   (EDCA backoff procedure).
    auto hasTransmitted = link.startTxop.has_value() && Simulator::Now() > *link.startTxop;

    m_queue->WipeAllExpiredMpdus();
    if ((hasTransmitted) ||
        (!m_queue->IsEmpty() && m_mac->GetChannelAccessManager(linkId)->GetGenerateBackoffOnNoTx()))
    {
        // GenerateBackoff(linkId);
        if (!m_queue->IsEmpty())
        {
            GenerateBackoff(linkId);
            Simulator::ScheduleNow(&ChannelAccessManager::RequestAccess,
                                   m_mac->GetChannelAccessManager(linkId),
                                   this);
        }
    }
    link.startTxop.reset();
    GetLink(linkId).access = NOT_REQUESTED;
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
