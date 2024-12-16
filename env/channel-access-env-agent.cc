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
    // if (m_moreData == false)
    // {
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
        env.nInQueuePackets = info.nInQueuePackets;
        env.nExpiredPackets = info.nExpiredPackets;
        env.nDroppedPackets = info.nDroppedPackets;
        env.nSendingPackets = info.nSendingPackets;
        env.nAckedPackets = info.nAckedPackets;
        env.nLowDelayPackets = info.nLowDelayPackets;
        env.nHighDelayPackets = info.nHighDelayPackets;

        // self link info
        auto selfUtil = sensingMac->GetSelfLinkChannelUtil(obs.period, obs.linkId);
        auto [txMean, txStd, txMin, txMax] = GetTxDataIntervalStats(selfUtil);
        auto traffics = sensingMac->GetSelfAppTraffic(obs.period);
        auto [appMean, appStd, appMin, appMax] = GetDataArrivalIntervalStats(traffics);
        SelfLinkObs selfLinkObs{
            .period = obs.period.ToDouble(Time::MS),
            .nTxDataPackets = static_cast<uint32_t>(selfUtil.size()),
            .nAckedDataPackets = GetNAckedDataPackets(selfUtil),
            .lastTxDataDelay = GetTxDataDelay(selfUtil),
            .lastTxOkDelay = GetTxOkDelay(selfUtil),
            .meanDataArrivalInterval = appMean,
            .stdDataArrivalInterval = appStd,
            .maxDataArrivalInterval = appMin,
            .minDataArrivalInterval = appMax,
            .meanTxDataInterval = txMean,
            .stdTxDataInterval = txStd,
            .maxTxDataInterval = txMin,
            .minTxDataInterval = txMax,
            .channelUtil = GetChannelUtil(selfUtil),
        };
        env.selfLinkObs = selfLinkObs;

        // reset neighbor counter
        m_idxNeighborSta = 0;
    // }

    // neighbor sta info
    auto neighborStas = sensingMac->GetNeighborMacAddress(obs.linkId);
    env.nNeighborStas = neighborStas.size();
    if (neighborStas.size() == 0)
    {
        return;
    }
    auto neighborUtil =
        sensingMac->GetNeighborChannelUtil(neighborStas[m_idxNeighborSta], obs.period);
    auto [nMean, nStd, nMin, nMax] = GetTxDataIntervalStats(neighborUtil);
    NeighborStaObs neighborStaObs{
        .role = static_cast<uint8_t>(sensingMac->GetNeighborStaRole(neighborStas[m_idxNeighborSta])),
        .period = obs.period.ToDouble(Time::MS),
        .nTxDataPackets = static_cast<uint32_t>(neighborUtil.size()),
        .nAckedDataPackets = GetNAckedDataPackets(neighborUtil),
        .lastTxDataDelay = GetTxDataDelay(neighborUtil),
        .lastTxOkDelay = GetTxOkDelay(neighborUtil),
        .meanTxDataInterval = nMean,
        .stdTxDataInterval = nStd,
        .maxTxDataInterval = nMin,
        .minTxDataInterval = nMax,
        .channelUtil = GetChannelUtil(neighborUtil),
        .meanRssi = GetMeanRssi(neighborUtil),
    };
    env.neighborStaObs = neighborStaObs;

    // update neighbor counter
    m_idxNeighborSta++;
    // m_moreData = (m_idxNeighborSta == neighborStas.size()) ? false : true;
    // env.moreData = m_moreData;
    // std::cout << "msg exchange: " << m_idxNeighborSta << "/" << neighborStas.size() << ", more data=" << m_moreData << std::endl;
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

uint32_t
ChannelAccessEnvAgent::GetNAckedDataPackets(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    uint32_t nTxDataPackets = 0;
    for (auto& packet : util)
    {
        if (packet.ackedTime != Seconds(0))
        {
            nTxDataPackets += 1;
        }
    }
    return nTxDataPackets;
}

double
ChannelAccessEnvAgent::GetTxDataDelay(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    if (util.size() == 0)
    {
        return Simulator::Now().ToDouble(Time::MS);
    }
    return (Simulator::Now() - util.back().txEndTime).ToDouble(Time::MS);
}

double
ChannelAccessEnvAgent::GetTxOkDelay(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    for (auto it = util.rbegin(); it != util.rend(); it++)
    {
        if (it->ackedTime != Seconds(0))
        {
            return (Simulator::Now() - it->ackedTime).ToDouble(Time::MS);
        }
    }
    return Simulator::Now().ToDouble(Time::MS);
}

std::array<double, 4>
ChannelAccessEnvAgent::GetDataArrivalIntervalStats(const std::vector<Time>& vec) const
{
    NS_LOG_FUNCTION(this);

    std::vector<double> intervals;
    for (auto it = vec.begin(); it != vec.end(); it++)
    {
        if (it != vec.begin())
        {
            double diff = (*it - *(it-1)).ToDouble(Time::MS);
            if (diff > 0)
            {
                intervals.push_back(diff);
            }
        }
    }
    return GetStats(intervals);
}

std::array<double, 4>
ChannelAccessEnvAgent::GetTxDataIntervalStats(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    std::vector<double> intervals;
    for (auto it = util.begin(); it != util.end(); it++)
    {
        if (it != util.begin())
        {
            double diff = (it->txEndTime - (it-1)->txEndTime).ToDouble(Time::MS);
            NS_ASSERT(diff > 0);
            intervals.push_back(diff);
        }
    }
    return GetStats(intervals);
}

double
ChannelAccessEnvAgent::GetChannelUtil(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    if (util.size() == 0)
    {
        return 0;
    }

    // Get the segements in the period
    std::vector<std::pair<Time, Time>> segments;
    for (const auto& packet : util)
    {
        auto begin = packet.txEndTime - packet.duration;
        segments.emplace_back(std::make_pair(begin, packet.txEndTime));
    }

    // Sort the segments by txTime
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Calculate channel utilization
    double channelUtil = 0;
    Time prevEnd = segments.front().first;
    for (const auto& [begin, end] : segments)
    {
        if (end <= prevEnd)
        {
            continue;
        }
        channelUtil += (end - std::max(begin, prevEnd)).ToDouble(Time::MS);
        prevEnd = end;
    }

    return channelUtil;
}

double
ChannelAccessEnvAgent::GetMeanRssi(const ChannelUtilList& util) const
{
    NS_LOG_FUNCTION(this);

    if (util.size() == 0)
    {
        return 0;
    }

    double sumRssi = 0;
    for (auto& packet : util)
    {
        sumRssi += packet.rssi;
    }
    return sumRssi / util.size();
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