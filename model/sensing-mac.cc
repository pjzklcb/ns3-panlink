/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "sensing-mac.h"

#include "ns3/env-utils.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-mac-queue.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("SensingMac");

NS_OBJECT_ENSURE_REGISTERED(SensingMac);

TypeId
SensingMac::GetTypeId()
{
    static TypeId tid = TypeId("ns3::SensingMac")
                            .SetParent<Object>()
                            .SetGroupName("Panlink")
                            .AddConstructor<SensingMac>()
                            .AddAttribute("MaxDuration",
                                          "The maximum duration of the data stored.",
                                          TimeValue(Seconds(1.0)),
                                          MakeTimeAccessor(&SensingMac::SetMaxDuration),
                                          MakeTimeChecker());
    return tid;
}

SensingMac::SensingMac()
{
    NS_LOG_FUNCTION(this);
}

SensingMac::~SensingMac()
{
    NS_LOG_FUNCTION(this);
}

void
SensingMac::DoInitialize()
{
    NS_LOG_FUNCTION(this);

    NS_ASSERT(m_mac);
    for (uint8_t linkId = 0; linkId < m_mac->GetNLinks(); ++linkId)
    {
        m_linkMap[linkId] = SelfLinkInfo();
    }
    for (const auto& [aci, ac] : wifiAcList)
    {
        m_queueMap[aci] = MacQueueInfo();
    }
}

void
SensingMac::DoDispose()
{
    NS_LOG_FUNCTION(this);

    m_linkMap.clear();
    m_neighborStaMap.clear();

    m_mac = nullptr;
}

void
SensingMac::SetWifiMac(const Ptr<WifiMac> mac)
{
    NS_LOG_FUNCTION(this << mac);
    m_mac = mac;
}

void
SensingMac::SetMaxDuration(Time duration)
{
    NS_LOG_FUNCTION(this << duration);
    m_duration = duration;
}

void
SensingMac::NotifyFrameInfo(const WifiMacHeader& hdr, double rssi, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << hdr << rssi << linkId);

    Mac48Address sender = hdr.GetAddr2();
    Mac48Address receiver = hdr.GetAddr1();

    // Recognize the role of the sender
    NeighborStaRole role = UNKNOWN;
    if (m_mac->GetTypeOfStation() == STA)
    {
        auto staMac = DynamicCast<PanlinkStaWifiMac>(m_mac);
        if (staMac->IsAssociated())
        {
            auto apAddr = staMac->GetLink(linkId).bssid;
            if (sender == apAddr)
            {
                role = ASSOCIATED_AP;
            }
            else if (receiver == apAddr)
            {
                role = BSS_STA;
            }
            else
            {
                role = OBSS;
            }
        }
    }

    auto [it, inserted] = m_neighborStaMap.try_emplace(sender);
    if (inserted)
    {
        it->second.linkId = linkId;
        it->second.role = role;
    }
    NS_ASSERT(it->second.linkId == linkId);

    if (role != UNKNOWN)
    {
        it->second.role = role;
    }
    it->second.channelUtil[Simulator::Now()] = hdr.GetDuration();
    it->second.rssi = rssi;

    auto& intervals = it->second.txIntervalList;
    if (intervals.size() >= 50)
    {
        double bin = GetBinOfTxIntervalHist(intervals.front());
        it->second.txIntervalHist[bin]--;
        if (it->second.txIntervalHist[bin] == 0)
        {
            it->second.txIntervalHist.erase(bin);
        }
        intervals.pop_front();
    }
    auto interval = Simulator::Now() - it->second.lastTxTime;
    intervals.push_back(interval);
    double bin = GetBinOfTxIntervalHist(interval);
    it->second.txIntervalHist[bin]++;
    it->second.lastTxTime = Simulator::Now();

    // Remove stale packets
    auto& util = m_neighborStaMap[sender].channelUtil;
    for (auto it = util.begin(); it != util.end();)
    {
        Time rxBegin = it->first - it->second;
        if (Simulator::Now() - rxBegin > m_duration)
        {
            it = util.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Remove inactive OBSS STAs
    for (auto it = m_neighborStaMap.begin(); it != m_neighborStaMap.end();)
    {
        if ((it->second.role == OBSS || it->second.role == UNKNOWN) &&
            it->second.channelUtil.size() == 0)
        {
            it = m_neighborStaMap.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void
SensingMac::NotifyLastTxOkSta(Mac48Address sta, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << sta << linkId);

    // Self packets are handled by NotifyAckedMpdu
    if (sta == m_mac->GetAddress())
    {
        return;
    }

    auto [it, inserted] = m_neighborStaMap.try_emplace(sta);
    if (inserted)
    {
        it->second.linkId = linkId;
    }
    NS_ASSERT(it->second.linkId == linkId);
    it->second.lastTxOkTime = Simulator::Now();
}

void
SensingMac::NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << ac << item);

    WipeAllExpiredPackets();

    if (!item->GetHeader().IsQosData())
    {
        return;
    }

    uint64_t uid = item->GetPacket()->GetUid();
    AcIndex aci = StringToAcIndex(ac);

    // register the packet
    auto [it, inserted] = m_packetMap.try_emplace(uid);
    NS_ABORT_MSG_IF(!inserted, "Packet with uid=" << uid << " already exists");

    it->second.packetSize = item->GetPacketSize();
    it->second.ac = aci;
    it->second.enqueueTime = Simulator::Now();
    it->second.duration = item->GetHeader().GetDuration();
}

void
SensingMac::NotifyEdcaExpired(std::string ac, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << ac << item);

    uint64_t uid = item->GetPacket()->GetUid();
    AcIndex aci = StringToAcIndex(ac);

    if (auto it = m_packetMap.find(uid); it != m_packetMap.end())
    {
        it->second.isExpired = true;

        // record expired packet
        m_queueMap[aci].expiredPackets.push_back(uid);
    }
}

void
SensingMac::NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW)
{
    NS_LOG_FUNCTION(this << context << p << txPowerW);

    auto linkId = static_cast<uint8_t>(std::stoi(context));
    if (auto it = m_packetMap.find(p->GetUid()); it != m_packetMap.end())
    {
        it->second.linkId = linkId;
        it->second.txCounter++;
        it->second.txPowerW = txPowerW;

        m_linkMap[linkId].pendingPackets.push_back(p->GetUid());
    }

    m_linkMap[linkId].channelUtil[p->GetUid()] = {Simulator::Now(), Seconds(0)};
}

void
SensingMac::NotifyPhyTxEnd(std::string context, Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << context << p);

    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));

    auto& util = m_linkMap[linkId].channelUtil;
    NS_ASSERT(util[p->GetUid()].second == Seconds(0));
    util[p->GetUid()].second = Simulator::Now();

    // Remove stale and invalid packets
    for (auto it = util.begin(); it != util.end();)
    {
        if (Simulator::Now() - it->second.first > m_duration || it->second.second == Seconds(0))
        {
            it = util.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void
SensingMac::NotifyAckedMpdu(Ptr<const WifiMpdu> mpdu)
{
    NS_LOG_FUNCTION(this << mpdu);

    auto uid = mpdu->GetPacket()->GetUid();
    if (auto it = m_packetMap.find(uid); it != m_packetMap.end())
    {
        it->second.ackedTime = Simulator::Now();
        m_linkMap[it->second.linkId].lastTxOkTime = Simulator::Now();
    }
}

void
SensingMac::NotifyDroppedMpdu(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu)
{
    NS_LOG_FUNCTION(this << reason << mpdu);

    // handle dropped packets reached retry limit or out of BA bitmap range
    if (reason != WIFI_MAC_DROP_REACHED_RETRY_LIMIT && reason != WIFI_MAC_DROP_QOS_OLD_PACKET)
    {
        return;
    }

    uint64_t uid = mpdu->GetPacket()->GetUid();
    if (auto it = m_packetMap.find(uid); it != m_packetMap.end())
    {
        it->second.isDropped = true;

        AcIndex aci = it->second.ac;
        m_queueMap[aci].droppedPackets.push_back(uid);
    }
}

void
SensingMac::WipeAllExpiredPackets()
{
    NS_LOG_FUNCTION(this);

    for (auto it = m_packetMap.begin(); it != m_packetMap.end();)
    {
        if (Simulator::Now() - it->second.enqueueTime > m_duration)
        {
            it = m_packetMap.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

double
SensingMac::GetThptReq(Time period, AcIndex ac, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << ac << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    double thptReq = 0;
    for (const auto& [uid, info] : m_packetMap)
    {
        if (info.ac == ac && info.linkId == linkId && info.enqueueTime >= Simulator::Now() - period)
        {
            thptReq += info.packetSize;
        }
    }

    return thptReq * 8e-6 / period.GetSeconds(); // Mbps
}

double
SensingMac::GetThpt(Time period, AcIndex ac, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << ac << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    double thpt = 0;
    for (const auto& [uid, info] : m_packetMap)
    {
        if (info.ac == ac && info.linkId == linkId &&
            info.enqueueTime >= Simulator::Now() - period && info.ackedTime != Seconds(0))
        {
            thpt += info.packetSize;
        }
    }

    return thpt * 8e-6 / period.GetSeconds(); // Mbps
}

double
SensingMac::GetDelay(Time period, AcIndex ac, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << ac << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    Time delay = Seconds(0);
    double counter = 0;
    for (const auto& [uid, info] : m_packetMap)
    {
        if (info.ac == ac && info.linkId == linkId &&
            info.enqueueTime >= Simulator::Now() - period && info.ackedTime != Seconds(0))
        {
            delay += info.ackedTime - info.enqueueTime;
            counter++;
        }
    }

    return counter == 0 ? 0 : delay.ToDouble(Time::MS) / counter; // ms
}

double
SensingMac::GetMinBssRssi(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    double rssi = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId && (sta.role == ASSOCIATED_AP || sta.role == BSS_STA))
        {
            rssi = std::min(rssi, sta.rssi);
        }
    }

    return rssi;
}

double
SensingMac::GetMaxObssRssi(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    double rssi = -101;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId && (sta.role == OBSS || sta.role == UNKNOWN))
        {
            rssi = std::max(rssi, sta.rssi);
        }
    }

    return rssi;
}

Time
SensingMac::GetSelfLastTxOk(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    auto linkIt = m_linkMap.find(linkId);
    NS_ASSERT_MSG(linkIt != m_linkMap.end(), "Link not found");

    return linkIt->second.lastTxOkTime;
}

uint32_t
SensingMac::GetNStaTxOkDelayLessThan(Time delay, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << delay << linkId);

    uint32_t nStas = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId)
        {
            if (Simulator::Now() - sta.lastTxOkTime <= delay)
            {
                nStas++;
            }
        }
    }

    return nStas;
}

double
SensingMac::GetNExpectedContentionStasAt(Time time, uint32_t granularity, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << time << linkId);

    double expectation = 0;
    for (auto& [sta, info] : m_neighborStaMap)
    {
        if (info.linkId != linkId || time < info.lastTxTime)
        {
            continue;
        }

        double bin = GetBinOfTxIntervalHist(time - info.lastTxTime);
        double prob = 0;
        for (auto [k, v] : info.txIntervalHist)
        {
            if (k >= bin - granularity && k <= bin + granularity)
            {
                prob += v / info.txIntervalList.size();
            }
        }

        double conditional_prob =
            GetProbOfTxIntervalLessThan(sta, time - info.lastTxTime);
        
        if (conditional_prob != 1)
        {
            expectation += prob / (1 - conditional_prob);
        }
        // std::cout << "MAC:" << sta << ", list=[";
        // for (auto [k, v] : info.txIntervalHist)
        // {
        //     std::cout << "[" << k << ", " << v << "], ";
        // }
        // std::cout << "last tx=" << info.lastTxTime.ToDouble(Time::MS) << "prob=" << prob <<
        // std::endl;
    }

    return expectation;
}

double
SensingMac::GetProbOfTxIntervalLessThan(Mac48Address sta, Time interval) const
{
    NS_LOG_FUNCTION(this << sta << interval);

    double prob = 0;
    if (auto it = m_neighborStaMap.find(sta); it != m_neighborStaMap.end())
    {
        if (it->second.txIntervalList.size() == 0)
        {
            return prob;
        }

        double counter = 0;
        for (auto in : it->second.txIntervalList)
        {
            if (in <= interval)
            {
                counter++;
            }
        }
        prob = counter / it->second.txIntervalList.size();
    }

    return prob;
}

double
SensingMac::GetBinOfTxIntervalHist(Time interval) const
{
    NS_LOG_FUNCTION(this << interval);

    double binWidth = 0.5; // ms
    return std::floor(interval.ToDouble(Time::MS) / binWidth);
}

uint32_t
SensingMac::GetNBssStas(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t nStas = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId && (sta.role == BSS_STA || sta.role == ASSOCIATED_AP))
        {
            nStas++;
        }
    }

    return nStas;
}

uint32_t
SensingMac::GetNObssStas(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t nStas = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId && (sta.role == OBSS || sta.role == UNKNOWN))
        {
            nStas++;
        }
    }

    return nStas;
}

double
SensingMac::GetSelfChannelUtil(Time period, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    auto linkIt = m_linkMap.find(linkId);
    NS_ASSERT_MSG(linkIt != m_linkMap.end(), "Link not found");

    // Get the segements in the period
    Time prevEnd = Simulator::Now() - period;
    double channelUtil = 0;
    for (const auto& [uid, pair] : linkIt->second.channelUtil)
    {
        Time begin = pair.first;
        Time end = (pair.second == Seconds(0)) ? Simulator::Now() : pair.second; // RX not end yet
        if (end <= prevEnd)
        {
            continue;
        }
        channelUtil += (end - std::max(prevEnd, begin)).GetSeconds();
        prevEnd = end;
    }

    return channelUtil / period.GetSeconds();
}

double
SensingMac::GetBssChannelUtil(Time period, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    // Get the segements in the period
    Time left = Simulator::Now() - period;
    Time right = Simulator::Now();
    std::vector<std::pair<Time, Time>> segments;
    for (const auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId != linkId || (sta.role != ASSOCIATED_AP && sta.role != BSS_STA))
        {
            continue;
        }

        for (auto [end, duration] : sta.channelUtil)
        {
            auto begin = end - duration;
            if (end <= left)
            {
                continue;
            }
            segments.emplace_back(std::make_pair(std::max(left, begin), std::min(right, end)));
        }
    }

    // Sort the segments by txTime
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Calculate channel utilization
    double channelUtil = 0;
    Time prevEnd = left;
    for (const auto& [begin, end] : segments)
    {
        if (end <= prevEnd)
        {
            continue;
        }
        channelUtil += (end - std::max(begin, prevEnd)).GetSeconds();
        prevEnd = end;
    }

    return channelUtil / period.GetSeconds();
}

double
SensingMac::GetOtherChannelUtil(Time period, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << linkId);

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    // Get the segements in the period
    Time left = Simulator::Now() - period;
    Time right = Simulator::Now();
    std::vector<std::pair<Time, Time>> segments;
    for (const auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId != linkId || sta.role == ASSOCIATED_AP || sta.role == BSS_STA)
        {
            continue;
        }

        for (auto [end, duration] : sta.channelUtil)
        {
            auto begin = end - duration;
            if (end <= left)
            {
                continue;
            }
            segments.emplace_back(std::make_pair(std::max(left, begin), std::min(right, end)));
        }
    }

    // Sort the segments by txTime
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Calculate channel utilization
    double channelUtil = 0;
    Time prevEnd = left;
    for (const auto& [begin, end] : segments)
    {
        if (end <= prevEnd)
        {
            continue;
        }
        channelUtil += (end - std::max(begin, prevEnd)).GetSeconds();
        prevEnd = end;
    }

    return channelUtil / period.GetSeconds();
}

SensingMac::AccessInfo
SensingMac::GetLastAccessInfo(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    AccessInfo result;
    for (auto& [aci, queue] : m_queueMap)
    {
        result.nInQueuePackets += m_mac->GetTxopQueue(aci)->GetNPackets();
        result.nDroppedPackets += queue.droppedPackets.size();
        result.nExpiredPackets += queue.expiredPackets.size();

        queue.droppedPackets.clear();
        queue.expiredPackets.clear();
    }

    result.nSendingPackets = m_linkMap[linkId].pendingPackets.size();
    for (const auto& uid : m_linkMap[linkId].pendingPackets)
    {
        auto it = m_packetMap.find(uid);
        NS_ASSERT(it != m_packetMap.end());
        if (it->second.ackedTime != Seconds(0))
        {
            result.nAckedPackets++;

            Time delay = it->second.ackedTime - it->second.enqueueTime;
            if (delay <= MilliSeconds(10))
            {
                result.nLowDelayPackets++;
            }
            else if (delay > MilliSeconds(30))
            {
                result.nHighDelayPackets++;
            }
        }
    }

    m_linkMap[linkId].pendingPackets.clear();

    return result;
}

} // namespace ns3
