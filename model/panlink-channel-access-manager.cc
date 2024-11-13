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
PanlinkChannelAccessManager::DoRestoreAccessIfNeeded()
{
    NS_LOG_FUNCTION(this);

    /**
     * Is there a Txop which needs to access the medium, and,
     * if there is one, when does backoff procedure restart (access restore)?
     */
    const auto accessGrant = GetAccessGrantStart();
    const auto now = Simulator::Now();
    bool needAccessRestore = false;
    for (auto txop : m_txops)
    {
        if (txop->GetAccessStatus(m_linkId) == Txop::REQUESTED)
        {
            auto aci = txop->GetWifiMacQueue()->GetAc();
            NS_LOG_DEBUG("expected access restore=" << accessGrant << " by " << aci);
            needAccessRestore = true;
        }
    }
    NS_LOG_DEBUG("Access restore needed: " << needAccessRestore);
    if (needAccessRestore)
    {
        auto expectedRestoreDelay = accessGrant - now;

        if (m_accessRestore.IsPending() &&
            Simulator::GetDelayLeft(m_accessRestore) < expectedRestoreDelay)
        {
            m_accessRestore.Cancel();
        }
        if (m_accessRestore.IsExpired())
        {
            m_accessRestore = Simulator::Schedule(expectedRestoreDelay,
                                                  &PanlinkChannelAccessManager::AccessRestore,
                                                  this);
        }
    }
}

void
PanlinkChannelAccessManager::AccessRestore()
{
    NS_LOG_FUNCTION(this);

    UpdateBackoff();

    // determine the txop that restores access
    for (auto txop : m_txops)
    {
        if (txop->GetAccessStatus(m_linkId) == Txop::REQUESTED)
        {
            StaticCast<PanlinkTxop>(txop)->GenerateNewBackoff(m_linkId);
            const auto aci = txop->GetWifiMacQueue()->GetAc();
            NS_LOG_DEBUG("access restored for " << aci);
        }
    }
    DoRestartAccessTimeoutIfNeeded();
}

void
PanlinkChannelAccessManager::DisableEdcaFor(Ptr<Txop> qosTxop, Time duration)
{
    NS_LOG_FUNCTION(this << qosTxop << duration);

    ChannelAccessManager::DisableEdcaFor(qosTxop, duration);
    DoRestoreAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyRxStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyRxStartNow(duration);
    DoRestoreAccessIfNeeded();
}

void 
PanlinkChannelAccessManager::NotifyTxStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyTxStartNow(duration);
    // wait until the access status changed
    Simulator::ScheduleNow(&PanlinkChannelAccessManager::DoRestoreAccessIfNeeded, this);
}

void
PanlinkChannelAccessManager::NotifyCcaBusyStartNow(Time duration,
                                                   WifiChannelListType channelType,
                                                   const std::vector<Time>& per20MhzDurations)
{
    NS_LOG_FUNCTION(this << duration << channelType);

    ChannelAccessManager::NotifyCcaBusyStartNow(duration, channelType, per20MhzDurations);
    DoRestoreAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyNavResetNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyNavResetNow(duration);
    DoRestoreAccessIfNeeded();
}

void
PanlinkChannelAccessManager::NotifyNavStartNow(Time duration)
{
    NS_LOG_FUNCTION(this << duration);

    ChannelAccessManager::NotifyNavStartNow(duration);
    DoRestoreAccessIfNeeded();
}

} // namespace ns3
