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
}

void
SensingMac::DoDispose()
{
    NS_LOG_FUNCTION(this);

    m_linkMap.clear();
    m_neighborStaMap.clear();

    if (m_mac)
    {
        m_mac->Dispose();
    }
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

    auto it = m_neighborStaMap.find(sender);
    if (it != m_neighborStaMap.end())
    {
        NS_ASSERT(it->second.linkId == linkId);

        if (it->second.role == UNKNOWN)
        {
            it->second.role = role;
        }
        else
        {
            NS_ASSERT(it->second.role == role);
        }
        it->second.channelUtil[Simulator::Now()] = hdr.GetDuration();
        it->second.rssi = rssi;
    }
    else
    {
        NeighborStaInfo info;
        info.role = role;
        info.linkId = linkId;
        info.channelUtil[Simulator::Now()] = hdr.GetDuration();
        info.rssi = rssi;
        m_neighborStaMap[sender] = info;
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

    auto it = m_neighborStaMap.find(sta);
    if (it != m_neighborStaMap.end())
    {
        NS_ASSERT(it->second.linkId == linkId);
        it->second.lastTxOkTime = Simulator::Now();
    }
    else
    {
        NeighborStaInfo info;
        info.linkId = linkId;
        info.lastTxOkTime = Simulator::Now();
        m_neighborStaMap[sta] = info;
    }
}

void
SensingMac::NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << ac << item);

    WipeAllExpiredPackets();

    auto uid = item->GetPacket()->GetUid();
    auto [it, inserted] = m_packetMap.try_emplace(uid);
    NS_ABORT_MSG_IF(!inserted, "Packet with uid=" << uid << " already exists");

    it->second.uid = uid;
    it->second.packetSize = item->GetPacketSize();
    it->second.ac = StringToAcIndex(ac);
    it->second.enqueueTime = Simulator::Now();
    it->second.duration = item->GetHeader().GetDuration();
}

void
SensingMac::NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW)
{
    NS_LOG_FUNCTION(this << context << p << txPowerW);

    auto linkId = static_cast<uint8_t>(std::stoi(context));
    auto it = m_packetMap.find(p->GetUid());
    if (it != m_packetMap.end())
    {
        it->second.linkId = linkId;
        it->second.phyTxTime.emplace_back(Simulator::Now());
        it->second.txPowerW = txPowerW;
        m_linkMap[linkId].packetList.emplace_back(p->GetUid());
    }
}

void
SensingMac::NotifyAckedMpdu(Ptr<const WifiMpdu> mpdu)
{
    NS_LOG_FUNCTION(this << mpdu);

    auto uid = mpdu->GetPacket()->GetUid();
    auto it = m_packetMap.find(uid);
    NS_ABORT_MSG_IF(it == m_packetMap.cend(), "Packet with uid=" << uid << " not found");

    it->second.ackedTime = Simulator::Now();

    m_linkMap[it->second.linkId].lastTxOkTime = Simulator::Now();
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

    auto linkIt = m_linkMap.find(linkId);
    NS_ASSERT_MSG(linkIt != m_linkMap.end(), "Link not found");

    double thptReq = 0;
    for (auto uid : linkIt->second.packetList)
    {
        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT_MSG(packetIt != m_packetMap.end(), " Packet not found");

        const auto& info = packetIt->second;
        if (info.ac == ac && Simulator::Now() - info.enqueueTime < period)
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

    auto linkIt = m_linkMap.find(linkId);
    NS_ASSERT_MSG(linkIt != m_linkMap.end(), "Link not found");

    double thpt = 0;
    for (const auto& uid : linkIt->second.packetList)
    {
        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT_MSG(packetIt != m_packetMap.end(), " Packet not found");

        const auto& info = packetIt->second;
        if (info.ac == ac && Simulator::Now() - info.enqueueTime < period &&
            info.ackedTime != Seconds(0))
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

    auto linkIt = m_linkMap.find(linkId);
    NS_ASSERT_MSG(linkIt != m_linkMap.end(), "Link not found");

    Time delay = Seconds(0);
    double counter = 0;
    for (const auto& uid : linkIt->second.packetList)
    {
        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT_MSG(packetIt != m_packetMap.end(), " Packet not found");

        const auto& info = packetIt->second;
        if (info.ac == ac && Simulator::Now() - info.enqueueTime < period &&
            info.ackedTime != Seconds(0))
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
        if (sta.linkId == linkId && sta.role == OBSS)
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

Time
SensingMac::GetOtherLastTxOk(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    Time lastTxOkTime = Seconds(0);
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId)
        {
            lastTxOkTime = std::max(lastTxOkTime, sta.lastTxOkTime);
        }
    }

    return lastTxOkTime;
}

uint32_t
SensingMac::GetNBssStas(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t nStas = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId && sta.role == BSS_STA)
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
        if (sta.linkId == linkId && sta.role == OBSS)
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
    Time left = Simulator::Now() - period;
    Time right = Simulator::Now();
    std::vector<std::pair<Time, Time>> segments;
    for (const auto& uid : linkIt->second.packetList)
    {
        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT_MSG(packetIt == m_packetMap.end(), "Packet not found");

        const auto& info = packetIt->second;
        for (auto txTime : info.phyTxTime)
        {
            auto rxTime = txTime + info.duration;
            if (rxTime < left)
            {
                continue;
            }
            segments.emplace_back(std::make_pair(std::max(left, txTime), std::min(right, rxTime)));
        }
    }

    // Sort the segments by txTime
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Calculate channel utilization
    double channelUtil = 0;
    Time prevRxTime = left;
    for (const auto& [begin, end] : segments)
    {
        if (end < prevRxTime)
        {
            continue;
        }
        channelUtil += (end - std::max(begin, prevRxTime)).GetSeconds();
        prevRxTime = end;
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
        if (sta.linkId != linkId)
        {
            continue;
        }

        for (auto [rxTime, duration] : sta.channelUtil)
        {
            auto txTime = rxTime + duration;
            if (rxTime < left)
            {
                continue;
            }
            segments.emplace_back(std::make_pair(std::max(left, txTime), std::min(right, rxTime)));
        }
    }

    // Sort the segments by txTime
    std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Calculate channel utilization
    double channelUtil = 0;
    Time prevRxTime = left;
    for (const auto& [begin, end] : segments)
    {
        if (end < prevRxTime)
        {
            continue;
        }
        channelUtil += (end - std::max(begin, prevRxTime)).GetSeconds();
        prevRxTime = end;
    }

    return channelUtil / period.GetSeconds();
}

} // namespace ns3
