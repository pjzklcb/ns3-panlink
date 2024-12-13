/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-channel-access-manager.h"

#include "panlink-txop.h"

#include "ns3/simulator.h"
#include "ns3/wifi-mac-queue.h"

#undef NS_LOG_APPEND_CONTEXT
#define NS_LOG_APPEND_CONTEXT std::clog << "[link=" << +m_linkId << "] "

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkChannelAccessManager");

NS_OBJECT_ENSURE_REGISTERED(PanlinkChannelAccessManager);

TypeId
PanlinkChannelAccessManager::GetTypeId()
{
    static TypeId tid = TypeId("ns3::PanlinkChannelAccessManager")
                            .SetParent<ChannelAccessManager>()
                            .SetGroupName("Panlink")
                            .AddConstructor<PanlinkChannelAccessManager>();
    return tid;
}

PanlinkChannelAccessManager::PanlinkChannelAccessManager()
{
    NS_LOG_FUNCTION(this);
}

PanlinkChannelAccessManager::~PanlinkChannelAccessManager()
{
    NS_LOG_FUNCTION(this);
}

void
PanlinkChannelAccessManager::DoResumeAccessIfNeeded()
{
    NS_LOG_FUNCTION(this);

    /**
     * Is there a Txop which needs to access the medium, and,
     * if there is one, when does backoff procedure restart (access resume)?
     */
    const auto accessGrant = GetAccessGrantStart();
    const auto now = Simulator::Now();
    bool needAccessResume = false;
    for (auto txop : m_txops)
    {
        if (txop->GetAccessStatus(m_linkId) == Txop::REQUESTED)
        {
            auto aci = txop->GetWifiMacQueue()->GetAc();
            NS_LOG_DEBUG("expected access resume=" << accessGrant << " by " << aci);
            needAccessResume = true;
        }
    }
    NS_LOG_DEBUG("Access resume needed: " << needAccessResume);
    if (needAccessResume)
    {
        /* Add one time step to the access resume delay to guarantee that the resume event won't 
        happen before the ACK transmission event, which will trigger NotifyTxStartNow.
        The expected time of resume event is in between [accessGrant, backoffStart], which won't 
        influence the update of backoffStart in UpdateBackoff method. */
        auto expectedResumeDelay = accessGrant - now + TimeStep(1);

        // Always set to the new access grant time as NAV may reset
        if (m_accessResume.IsPending())
        {
            m_accessResume.Cancel();
        }
        if (m_accessResume.IsExpired())
        {
            m_accessResume = Simulator::Schedule(expectedResumeDelay,
                                                  &PanlinkChannelAccessManager::AccessResume,
                                                  this);
        }
    }
}

void
PanlinkChannelAccessManager::AccessResume()
{
    NS_LOG_FUNCTION(this);

    UpdateBackoff();

    // determine the txop that resumes access
    bool needAccessResume = false;
    for (auto txop : m_txops)
    {
        if (txop->GetAccessStatus(m_linkId) == Txop::REQUESTED)
        {
            needAccessResume = true;
            StaticCast<PanlinkTxop>(txop)->GenerateNewBackoff(m_linkId);
            const auto aci = txop->GetWifiMacQueue()->GetAc();
            NS_LOG_DEBUG("access resumed for " << aci);
        }
    }
    NS_ASSERT(needAccessResume);
    DoRestartAccessTimeoutIfNeeded();
}

void
PanlinkChannelAccessManager::DisableEdcaFor(Ptr<Txop> qosTxop, Time duration)
{
    NS_LOG_FUNCTION(this << qosTxop << duration);

    ChannelAccessManager::DisableEdcaFor(qosTxop, duration);
    DoResumeAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyRxStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyRxStartNow(duration);
    DoResumeAccessIfNeeded();
}

void 
PanlinkChannelAccessManager::NotifyTxStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyTxStartNow(duration);
    // wait until the access status changed
    Simulator::ScheduleNow(&PanlinkChannelAccessManager::DoResumeAccessIfNeeded, this);
}

void
PanlinkChannelAccessManager::NotifyCcaBusyStartNow(Time duration,
                                                   WifiChannelListType channelType,
                                                   const std::vector<Time>& per20MhzDurations)
{
    NS_LOG_FUNCTION(this << duration << channelType);

    ChannelAccessManager::NotifyCcaBusyStartNow(duration, channelType, per20MhzDurations);
    DoResumeAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyNavResetNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyNavResetNow(duration);
    DoResumeAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyNavStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyNavStartNow(duration);
    DoResumeAccessIfNeeded();
}

} // namespace ns3
