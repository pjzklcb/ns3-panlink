/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "channel-access-env-agent.h"

#include "ns3/panlink-txop.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ChannelAccessEnvAgent");

NS_OBJECT_ENSURE_REGISTERED(ChannelAccessEnvAgent);

TypeId
ChannelAccessEnvAgent::GetTypeId()
{
    static TypeId tid = TypeId("ns3::ChannelAccessEnvAgent")
                            .SetParent<EnvAgent>()
                            .SetGroupName("Panlink")
                            .AddConstructor<ChannelAccessEnvAgent>();
    return tid;
}

ChannelAccessEnvAgent::ChannelAccessEnvAgent()
{
    NS_LOG_FUNCTION(this);
    m_type = RL_CHANNEL_ACCESS;
}

ChannelAccessEnvAgent::~ChannelAccessEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

void
ChannelAccessEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    const auto now = Simulator::Now();

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.agentType = m_type;

    // Get obs from sensing MAC
    auto sensingMac = GetSensingMac();
    if (m_moreData == false)
    {
        // ideal obs
        env.nStasWithPackets = sensingMac->GetNStasWithPacketsNow(obs.linkId);
        env.minNeighborBackoff = sensingMac->GetMinNeighborBackoff(obs.linkId);

        // obs from channel access
        auto& channelAccessObs = static_cast<ChannelAccessObs&>(obs);
        env.ac = static_cast<uint8_t>(channelAccessObs.ac);
        env.cw = channelAccessObs.cw;
        env.nDeferredAccess = channelAccessObs.nDeferredAccess;
        env.waitedSlots = channelAccessObs.waitedSlots;

        // access info
        auto info = sensingMac->GetLastAccessInfo(obs.linkId);
        LastAccessObs lastAccessObs{.nInQueuePackets = info.nInQueuePackets,
                                    .nExpiredPackets = info.nExpiredPackets,
                                    .nDroppedPackets = info.nDroppedPackets,
                                    .nSendingPackets = info.nSendingPackets,
                                    .nAckedPackets = info.nAckedPackets,
                                    .nLowDelayPackets = info.nLowDelayPackets,
                                    .nHighDelayPackets = info.nHighDelayPackets};
        env.lastAccessObs = lastAccessObs;

        // self link info
        auto selfTxData = sensingMac->GetSeltLinkTxDataTime(obs.period, obs.linkId);
        auto selfAcks = sensingMac->GetSelfLinkAckedTime(obs.period, obs.linkId);
        auto traffics = sensingMac->GetSelfAppTraffic(obs.period);
        auto rxFails = sensingMac->GetRxFailTime(obs.period, obs.linkId);
        SelfLinkObs selfLinkObs{
            .period = obs.period.ToDouble(Time::MS),
            .nTxDataPackets = static_cast<uint32_t>(selfTxData.size()),
            .nAckedDataPackets = static_cast<uint32_t>(selfAcks.size()),
            .lastTxOkDelay = GetLastDelay(selfAcks),
            .lastRxFailDelay = GetLastDelay(rxFails),
            .lastRxFailIdleTime = sensingMac->GetLastRxFailIdleTime(obs.linkId),
            .nRxFail = static_cast<uint32_t>(rxFails.size()),
            .idleRatio = sensingMac->GetChannelIdleRatio(obs.period, obs.linkId),
            .trafficIntervalStats = GetIntervalStats(traffics),
            .txDataIntervalStats = GetIntervalStats(selfTxData)};
        env.selfLinkObs = selfLinkObs;

        // reset neighbor counter
        m_idxNeighborSta = 0;
    }

    // neighbor sta info
    auto neighborStas = sensingMac->GetNeighborMacAddress(obs.linkId);
    env.nNeighborStas = neighborStas.size();
    if (neighborStas.size() == 0)
    {
        return;
    }
    auto neighborAcks =
        sensingMac->GetNeighborAckedTime(neighborStas[m_idxNeighborSta], obs.period);
    NeighborStaObs neighborStaObs{
        .role =
            static_cast<uint8_t>(sensingMac->GetNeighborStaRole(neighborStas[m_idxNeighborSta])),
        .period = obs.period.ToDouble(Time::MS),
        .nAcks = static_cast<uint32_t>(neighborAcks.size()),
        .lastTxOkDelay = GetLastDelay(neighborAcks),
        .lastTxOkIdleTime = sensingMac->GetNeighborLastTxOkIdleTime(neighborStas[m_idxNeighborSta]),
        .meanRssi = sensingMac->GetNeighborMeanRssi(neighborStas[m_idxNeighborSta], obs.period),
        .txOkIntervalStats = GetIntervalStats(neighborAcks),
        .backoffSlots = sensingMac->GetNeighborBackoff(neighborStas[m_idxNeighborSta], obs.linkId),
        .cw = sensingMac->GetNeighborCw(neighborStas[m_idxNeighborSta], obs.linkId)};
    env.neighborStaObs = neighborStaObs;

    // update more data flag
    m_idxNeighborSta++;
    m_moreData = (m_idxNeighborSta == neighborStas.size()) ? false : true;
    env.moreData = m_moreData;
    NS_LOG_DEBUG("msg exchange: " << m_idxNeighborSta << "/" << neighborStas.size()
                                  << ", more data=" << m_moreData);
}

void
ChannelAccessEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    auto& act = msgInterface->GetPy2CppVector()->at(m_memoryId);
    double newBackoff = act.backoff;

    AcIndex ac = static_cast<ChannelAccessObs&>(obs).ac;
    auto txop = StaticCast<PanlinkTxop>(GetWifiMac()->GetQosTxop(ac));
    if (txop->GetBackoffSlots(obs.linkId) != 0)
    {
        NS_LOG_DEBUG("reset backoff from " << txop->GetBackoffSlots(obs.linkId) << " to "
                                           << newBackoff << " slots");
    }
    else
    {
        NS_LOG_DEBUG("start backoff=" << newBackoff << " slots");
    }
    txop->SetBackoffSlots(newBackoff, obs.linkId);
}

double
ChannelAccessEnvAgent::GetLastDelay(const std::vector<Time>& vec) const
{
    NS_LOG_FUNCTION(this);

    if (!vec.empty())
    {
        return (Simulator::Now() - vec.back()).ToDouble(Time::MS);
    }
    return Simulator::Now().ToDouble(Time::MS);
}

// double
// ChannelAccessEnvAgent::GetChannelUtil(const std::vector<Time>& util) const
// {
//     NS_LOG_FUNCTION(this);

//     if (util.size() == 0)
//     {
//         return 0;
//     }

//     // Get the segements in the period
//     std::vector<std::pair<Time, Time>> segments;
//     for (const auto& packet : util)
//     {
//         auto begin = packet.txEndTime - packet.duration;
//         segments.emplace_back(std::make_pair(begin, packet.txEndTime));
//     }

//     // Sort the segments by txTime
//     std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
//         return a.first < b.first;
//     });

//     // Calculate channel utilization
//     double channelUtil = 0;
//     Time prevEnd = segments.front().first;
//     for (const auto& [begin, end] : segments)
//     {
//         if (end <= prevEnd)
//         {
//             continue;
//         }
//         channelUtil += (end - std::max(begin, prevEnd)).ToDouble(Time::MS);
//         prevEnd = end;
//     }

//     return channelUtil;
// }

std::array<double, 4>
ChannelAccessEnvAgent::GetIntervalStats(const std::vector<Time>& vec) const
{
    NS_LOG_FUNCTION(this);

    std::vector<double> intervals;
    for (auto it = vec.begin(); it != vec.end(); it++)
    {
        if (it != vec.begin())
        {
            double diff = (*it - *(it - 1)).ToDouble(Time::MS);
            if (diff > 0)
            {
                intervals.push_back(diff);
            }
        }
    }
    return GetStats(intervals);
}

std::array<double, 4>
ChannelAccessEnvAgent::GetStats(const std::vector<double>& vec) const
{
    NS_LOG_FUNCTION(this);

    std::array<double, 4> stats;
    stats.fill(0);
    if (vec.size() == 0)
    {
        return stats;
    }
    stats[0] = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
    double sumOfSquaredDifferences =
        std::accumulate(vec.begin(), vec.end(), 0.0, [mean = stats[0]](double acc, double x) {
            return acc + (x - mean) * (x - mean);
        });
    stats[1] = (vec.size() > 1) ? std::sqrt(sumOfSquaredDifferences / (vec.size() - 1)) : 0;
    stats[2] = *std::min_element(vec.begin(), vec.end());
    stats[3] = *std::max_element(vec.begin(), vec.end());
    return stats;
}

} // namespace ns3