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
     * if there is one, when does its backoff countdown restart (access restore)?
     */
    Ptr<Txop> nextTxop;
    auto expectedBackoffStart = Simulator::GetMaximumSimulationTime();
    const auto now = Simulator::Now();
    for (auto txop : m_txops)
    {
        auto panlinkTxop = StaticCast<PanlinkTxop>(txop);
        NS_ASSERT(panlinkTxop->GetLink(m_linkId).backoffStart >= now);

        if (txop->GetAccessStatus(m_linkId) == Txop::REQUESTED)
        {
            if (auto backoffStart = GetBackoffStartFor(txop);
                backoffStart > now && backoffStart < expectedBackoffStart)
            {
                expectedBackoffStart = backoffStart;
                nextTxop = txop;
            }
        }
    }
    NS_LOG_DEBUG("Access restore needed: " << (nextTxop != nullptr));
    if (nextTxop)
    {
        const auto aci = nextTxop->GetWifiMacQueue()->GetAc();
        NS_LOG_DEBUG("expected access restore=" << expectedBackoffStart << " by " << aci);
        auto expectedRestoreDelay = expectedBackoffStart - now;

        if (m_accessRestore.IsPending() &&
            Simulator::GetDelayLeft(m_accessRestore) > expectedRestoreDelay)
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
            if (GetBackoffStartFor(txop) == Simulator::Now())
            {
                StaticCast<PanlinkTxop>(txop)->GenerateNewBackoff(m_linkId);
                const auto aci = txop->GetWifiMacQueue()->GetAc();
                NS_LOG_DEBUG("access restored by " << aci);
            }
        }
    }

    DoRestartAccessTimeoutIfNeeded();
}

} // namespace ns3
