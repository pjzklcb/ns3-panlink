/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "sensing-mac.h"

#include "ns3/env-utils.h"
#include "ns3/node-list.h"
#include "ns3/node.h"
#include "ns3/qos-txop.h"
#include "ns3/wifi-mac-queue.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-net-device.h"

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
        m_queueMap[aci] = SelfMacQueueInfo();
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
SensingMac::NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << ac << item);

    WipeAllStaledPackets();

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

    m_queueMap[aci].traffics.push_back(uid);
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
        it->second.txPowerW = txPowerW;
        it->second.phyTxTime.push_back({Simulator::Now(), Seconds(0)});

        m_linkMap[linkId].pendingPackets.push_back(p->GetUid());
    }
}

void
SensingMac::NotifyPhyTxEnd(std::string context, Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << context << p);

    Time now = Simulator::Now();
    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));
    if (auto it = m_packetMap.find(p->GetUid()); it != m_packetMap.end())
    {
        // check if the packet is not recorded in PhyTxBegin
        NS_ASSERT(!it->second.phyTxTime.empty() &&
                  it->second.phyTxTime.back().second == Seconds(0));
        it->second.phyTxTime.back().second = now;
        m_linkMap[linkId].channelUtil.push_back({p->GetUid(), now});
    }

    // Remove staled and invalid tx packets.
    while (!m_linkMap[linkId].channelUtil.empty())
    {
        auto [_, txEndTime] = m_linkMap[linkId].channelUtil.front();
        if (now - txEndTime > m_duration)
        {
            m_linkMap[linkId].channelUtil.pop_front();
        }
        else
        {
            break;
        }
    }
}

void
SensingMac::NotifyPhyRxPpduDrop(std::string context,
                                Ptr<const WifiPpdu> ppdu,
                                WifiPhyRxfailureReason reason)
{
    NS_LOG_FUNCTION(this << context << ppdu << reason);

    if (reason == UNSUPPORTED_SETTINGS || reason == CHANNEL_SWITCHING || reason == SLEEPING ||
        reason == POWERED_OFF)
    {
        return;
    }
    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));
    m_linkMap[linkId].rxFailTime.push_back(Simulator::Now());
}

void
SensingMac::NotifyPhyState(std::string context, Time begin, Time duration, WifiPhyState state)
{
    NS_LOG_FUNCTION(this << context << begin << duration << state);

    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));
    if (state != WifiPhyState::IDLE)
    {
        return;
    }
    m_linkMap[linkId].idleTime.push_back({begin, duration});
}

void
SensingMac::NotifyAckedMpdu(Ptr<const WifiMpdu> mpdu)
{
    NS_LOG_FUNCTION(this << mpdu);

    auto uid = mpdu->GetPacket()->GetUid();
    if (auto it = m_packetMap.find(uid); it != m_packetMap.end())
    {
        it->second.ackedTime = Simulator::Now();
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
SensingMac::WipeAllStaledPackets()
{
    NS_LOG_FUNCTION(this);

    // wipe self packets
    for (auto it = m_packetMap.begin(); it != m_packetMap.end();)
    {
        if (Simulator::Now() - it->second.enqueueTime > m_duration)
        {
            // remove from mac queue traffic
            auto& d = m_queueMap[it->second.ac].traffics;
            auto queueIt = std::find(d.begin(), d.end(), it->first);
            NS_ASSERT(queueIt != d.end());
            d.erase(queueIt);

            it = m_packetMap.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // wipe neighbor packets
    for (auto it = m_neighborPacketMap.begin(); it != m_neighborPacketMap.end();)
    {
        if (Simulator::Now() - it->second.phyRxTime.back().first > m_duration)
        {
            it = m_neighborPacketMap.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // Remove inactive STAs
    WipeAllInactiveNeighborStas();
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

uint32_t
SensingMac::GetNRxFail(Time period, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << period << linkId);

    Time now = Simulator::Now();
    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    // eliminate staled rx fails
    while (!m_linkMap[linkId].rxFailTime.empty())
    {
        auto fail = m_linkMap[linkId].rxFailTime.front();
        if (now - fail > m_duration)
        {
            m_linkMap[linkId].rxFailTime.pop_front();
        }
        else
        {
            break;
        }
    }

    uint32_t nRxFail = 0;
    for (const auto& failTime : m_linkMap[linkId].rxFailTime)
    {
        if (now - failTime <= period)
        {
            nRxFail++;
        }
    }
    return nRxFail;
}

double
SensingMac::GetChannelIdleRatio(Time period, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << period << linkId);

    Time now = Simulator::Now();
    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    // eliminate staled idle period
    while (!m_linkMap[linkId].idleTime.empty())
    {
        auto idle = m_linkMap[linkId].idleTime.front();
        if (now - (idle.first + idle.second) > m_duration)
        {
            m_linkMap[linkId].idleTime.pop_front();
        }
        else
        {
            break;
        }
    }

    double idleTime = 0;
    for (const auto& [begin, duration] : m_linkMap[linkId].idleTime)
    {
        Time end = begin + duration;
        if (now - end <= period)
        {
            idleTime += (end - std::max(now - period, begin)).ToDouble(Time::MS);
        }
    }
    return idleTime / period.ToDouble(Time::MS);
}

void
SensingMac::NotifyPhyRxBegin(std::string context,
                             Ptr<const Packet> p,
                             RxPowerWattPerChannelBand rxPowersW)
{
    NS_LOG_FUNCTION(this << context << p);

    Time now = Simulator::Now();
    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));
    uint64_t uid = p->GetUid();

    // create neighbor packet info
    auto [it, inserted] = m_neighborPacketMap.try_emplace(uid);
    if (inserted)
    {
        it->second.linkId = linkId;
    }
    NS_ASSERT(it->second.linkId == linkId);
    it->second.phyRxTime.push_back({now, Seconds(0)});
}

void
SensingMac::NotifyPhyRxEnd(std::string context, Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << context << p);

    Time now = Simulator::Now();
    uint8_t linkId = static_cast<uint8_t>(std::stoi(context));
    uint64_t uid = p->GetUid();

    auto it = m_neighborPacketMap.find(uid);
    if (it == m_neighborPacketMap.end())
    {
        return;
    }
    NS_ASSERT(it->second.linkId == linkId && it->second.phyRxTime.back().second == Seconds(0));
    it->second.phyRxTime.back().second = now;
}

void
SensingMac::NotifyFrameInfo(Ptr<const WifiMpdu> mpdu,
                            const WifiMacHeader& hdr,
                            double rssi,
                            uint8_t linkId)
{
    NS_LOG_FUNCTION(this << hdr << rssi << linkId);

    Time now = Simulator::Now();
    Mac48Address sender = hdr.GetAddr2();
    Mac48Address receiver = hdr.GetAddr1();
    uint64_t uid = mpdu->GetPacket()->GetUid();

    // recognize the role of the sender
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

    // update neighbor sta info
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
    it->second.channelUtil.push_back({uid, now});

    // update neighbor packet info
    auto packetIt = m_neighborPacketMap.find(uid);
    NS_ASSERT(packetIt != m_neighborPacketMap.end() && packetIt->second.linkId == linkId &&
              packetIt->second.phyRxTime.back().second == now);
    packetIt->second.isData = hdr.IsData();
    packetIt->second.rssi = rssi;
    if (role == ASSOCIATED_AP && receiver == m_mac->GetAddress())
    {
        // if the packet is from AP to this MAC, assume it will be acked
        packetIt->second.ackedTime = now + MicroSeconds(50);
    }

    // Remove staled and invalid rx packets.
    while (!it->second.channelUtil.empty())
    {
        auto [_, rxEndTime] = it->second.channelUtil.front();
        if (now - rxEndTime > m_duration)
        {
            it->second.channelUtil.pop_front();
        }
        else
        {
            break;
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

    auto it = m_neighborStaMap.find(sta);
    if (it == m_neighborStaMap.end())
    {
        return;
    }
    NS_ASSERT(it->second.linkId == linkId);

    Time now = Simulator::Now();
    auto [lastUid, rxEndTime] = it->second.channelUtil.back();
    if (auto packetIt = m_neighborPacketMap.find(lastUid); packetIt != m_neighborPacketMap.end())
    {
        auto& lastPacket = packetIt->second;
        /**
         * @todo: ack timeout is based on PHY preamble used by the neighbor STA.
         * Also, need a better way to check collision
         */
        if (lastPacket.ackedTime == Seconds(0) && now - rxEndTime < MicroSeconds(70))
        {
            lastPacket.ackedTime = now;
        }
    }

    it->second.ackedTime.push_back(now);

    // remove staled acked time
    while (!it->second.ackedTime.empty())
    {
        auto ackedTime = it->second.ackedTime.front();
        if (now - ackedTime > m_duration)
        {
            it->second.ackedTime.pop_front();
        }
        else
        {
            break;
        }
    }
}

void
SensingMac::WipeAllInactiveNeighborStas()
{
    NS_LOG_FUNCTION(this);

    for (auto it = m_neighborStaMap.begin(); it != m_neighborStaMap.end();)
    {
        if (it->second.channelUtil.size() == 0)
        {
            it = m_neighborStaMap.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

uint32_t
SensingMac::GetNStasWithPacketsNow(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t nStas = 0;
    for (uint32_t i = 0; i < NodeList::GetNNodes(); i++)
    {
        Ptr<Node> node = NodeList::GetNode(i);
        Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice>(node->GetDevice(0));
        Ptr<WifiMac> mac = device->GetMac();
        if (mac->GetAddress() == m_mac->GetAddress())
        {
            continue;
        }
        auto it = m_neighborStaMap.find(mac->GetAddress());
        if (it == m_neighborStaMap.end() || it->second.linkId != linkId)
        {
            continue;
        }

        int counter = 0;
        for (auto [aci, _] : wifiAcList)
        {
            counter += mac->GetTxopQueue(aci)->GetNPackets();
        }
        if (counter > 0)
        {
            nStas++;
        }
    }

    return nStas;
}

uint32_t
SensingMac::GetMinNeighborBackoff(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t backoff = 1024;
    for (uint32_t i = 0; i < NodeList::GetNNodes(); i++)
    {
        Ptr<Node> node = NodeList::GetNode(i);
        Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice>(node->GetDevice(0));
        Ptr<WifiMac> mac = device->GetMac();
        if (mac->GetAddress() == m_mac->GetAddress())
        {
            continue;
        }
        auto it = m_neighborStaMap.find(mac->GetAddress());
        if (it == m_neighborStaMap.end() || it->second.linkId != linkId)
        {
            continue;
        }

        uint32_t staBackoff = 1024;
        if (auto queue = mac->GetTxopQueue(AC_BE_NQOS);
            queue != nullptr && queue->GetNPackets() > 0)
        {
            staBackoff = std::min(staBackoff, mac->GetTxop()->GetBackoffSlots(linkId));
        }
        for (auto [aci, _] : wifiAcList)
        {
            if (mac->GetTxopQueue(aci)->GetNPackets() > 0)
            {
                staBackoff = std::min(staBackoff, mac->GetQosTxop(aci)->GetBackoffSlots(linkId));
            }
        }
        // if (staBackoff != 1024)
        // {
        //     std::cout << "STA: " << mac->GetAddress() << ", backoff=" << staBackoff << std::endl;
        // }
        backoff = std::min(backoff, staBackoff);
    }

    return backoff;
}

uint32_t
SensingMac::GetNeighborBackoff(Mac48Address sta, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << sta << linkId);

    uint32_t backoff = 1024;
    for (uint32_t i = 0; i < NodeList::GetNNodes(); i++)
    {
        Ptr<Node> node = NodeList::GetNode(i);
        Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice>(node->GetDevice(0));
        Ptr<WifiMac> mac = device->GetMac();
        if (mac->GetAddress() == sta)
        {
            if (auto queue = mac->GetTxopQueue(AC_BE_NQOS);
                queue != nullptr && queue->GetNPackets() > 0)
            {
                backoff = std::min(backoff, mac->GetTxop()->GetBackoffSlots(linkId));
            }
            for (auto [aci, _] : wifiAcList)
            {
                if (mac->GetTxopQueue(aci)->GetNPackets() > 0)
                {
                    backoff = std::min(backoff, mac->GetQosTxop(aci)->GetBackoffSlots(linkId));
                }
            }
            break;
        }
    }

    return backoff;
}

std::array<uint32_t, 4>
SensingMac::GetNeighborCw(Mac48Address sta, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << sta << linkId);

    std::array<uint32_t, 4> cw;
    for (uint32_t i = 0; i < NodeList::GetNNodes(); i++)
    {
        Ptr<Node> node = NodeList::GetNode(i);
        Ptr<WifiNetDevice> device = DynamicCast<WifiNetDevice>(node->GetDevice(0));
        Ptr<WifiMac> mac = device->GetMac();
        if (mac->GetAddress() == sta)
        {
            int i = 0;
            for (auto [aci, _] : wifiAcList)
            {
                cw[i++] = mac->GetQosTxop(aci)->GetCw(linkId);
            }
            break;
        }
    }

    return cw;
}

uint32_t
SensingMac::GetNNeighborStas(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    uint32_t nStas = 0;
    for (auto& [_, sta] : m_neighborStaMap)
    {
        if (sta.linkId == linkId)
        {
            nStas++;
        }
    }

    return nStas;
}

SensingMac::NeighborStaRole
SensingMac::GetNeighborStaRole(Mac48Address sta) const
{
    NS_LOG_FUNCTION(this << sta);

    auto it = m_neighborStaMap.find(sta);
    NS_ASSERT_MSG(it != m_neighborStaMap.end(), "STA not found");
    return it->second.role;
}

std::vector<Time>
SensingMac::GetSeltLinkTxDataTime(Time period, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << linkId);

    auto it = m_linkMap.find(linkId);
    NS_ASSERT_MSG(it != m_linkMap.end(), "Link not found");

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    std::vector<Time> result;
    Time prevEnd = Simulator::Now() - period;
    for (const auto& [uid, txEndTime] : it->second.channelUtil)
    {
        if (txEndTime <= prevEnd) // skip packet not in the period
        {
            continue;
        }

        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT(packetIt != m_packetMap.end());

        result.push_back(txEndTime);

        // make sure AMPDU with same tx end time will be ignored
        prevEnd = txEndTime;
    }

    return result;
}

std::vector<Time>
SensingMac::GetSelfLinkAckedTime(Time period, uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << period << linkId);

    auto it = m_linkMap.find(linkId);
    NS_ASSERT_MSG(it != m_linkMap.end(), "Link not found");

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    std::vector<Time> result;
    Time prevEnd = Simulator::Now() - period;
    for (const auto& [uid, txEndTime] : it->second.channelUtil)
    {
        if (txEndTime <= prevEnd) // skip packets not in the period
        {
            continue;
        }
        auto packetIt = m_packetMap.find(uid);
        NS_ASSERT(packetIt != m_packetMap.end());

        // find this tx in the phyTxTime
        auto vec = packetIt->second.phyTxTime;
        auto phyTxIt = std::find_if(vec.begin(), vec.end(), [txEndTime](const auto& pair) {
            return pair.second == txEndTime;
        });
        // get the duration of this tx
        // Time duration = phyTxIt->second - phyTxIt->first;
        // only the last tx can use acked time
        // Time ackedTime =
        //     (std::next(phyTxIt) == vec.end()) ? packetIt->second.ackedTime : Seconds(0);

        // ChannelUtilInfo packetInfo{.duration = duration, // @todo: without SIFS and ack time
        //                            .txEndTime = txEndTime,
        //                            .ackedTime = ackedTime};
        if (std::next(phyTxIt) == vec.end() && packetIt->second.ackedTime > prevEnd)
        {
            result.push_back(packetIt->second.ackedTime);
        }

        // make sure AMPDU with same tx end time will be ignored
        prevEnd = txEndTime;
    }

    return result;
}

std::vector<Time>
SensingMac::GetSelfAppTraffic(Time period) const
{
    NS_LOG_FUNCTION(this << period);

    std::vector<Time> result;
    for (const auto& [aci, _] : wifiAcList)
    {
        for (auto uid : m_queueMap.at(aci).traffics)
        {
            auto it = m_packetMap.find(uid);
            if (it == m_packetMap.end() || Simulator::Now() - it->second.enqueueTime > period)
            {
                continue;
            }
            result.push_back(it->second.enqueueTime);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

double
SensingMac::GetNeighborMeanRssi(Mac48Address sta, Time period) const
{
    NS_LOG_FUNCTION(this << sta << period);

    auto it = m_neighborStaMap.find(sta);
    NS_ASSERT_MSG(it != m_neighborStaMap.end(), "STA not found");

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    uint32_t nTx = 0;
    double rssi = 0;
    Time prevEnd = Simulator::Now() - period;
    for (const auto& [uid, rxEndTime] : it->second.channelUtil)
    {
        auto packetIt = m_neighborPacketMap.find(uid);
        NS_ASSERT(packetIt != m_neighborPacketMap.end());

        // skip those not in the period
        if (rxEndTime <= prevEnd)
        {
            continue;
        }

        nTx++;
        NS_ASSERT(packetIt->second.rssi <= -10);
        rssi += packetIt->second.rssi;
        // make sure AMPDU with same rx end time will be ignored
        prevEnd = rxEndTime;
    }

    return rssi / nTx;
}

double
SensingMac::GetNeighborLastTxOkIdleTime(Mac48Address sta) const
{
    NS_LOG_FUNCTION(this << sta);

    auto it = m_neighborStaMap.find(sta);
    NS_ASSERT_MSG(it != m_neighborStaMap.end(), "STA not found");

    if (it->second.ackedTime.empty())
    {
        return 0;
    }

    Time lastTxOk = it->second.ackedTime.back();
    double result = 0;
    for (const auto& [begin, duration] : m_linkMap.at(it->second.linkId).idleTime)
    {
        if (lastTxOk > begin)
        {
            continue;
        }

        result += duration.ToDouble(Time::MS);
    }

    return result;
}

std::vector<Time>
SensingMac::GetNeighborAckedTime(Mac48Address sta, Time period) const
{
    NS_LOG_FUNCTION(this << sta << period);

    auto it = m_neighborStaMap.find(sta);
    NS_ASSERT_MSG(it != m_neighborStaMap.end(), "STA not found");

    NS_ASSERT(period > Seconds(0));
    NS_ASSERT_MSG(period <= m_duration, "Period surpass maximum duration");

    std::vector<Time> result;
    Time prevEnd = Simulator::Now() - period;
    for (const auto& ackedTime : it->second.ackedTime)
    {
        // skip those not in the period
        if (ackedTime <= prevEnd)
        {
            continue;
        }

        // // find this rx in the phyRxTime
        // auto vec = packetIt->second.phyRxTime;
        // auto phyRxIt = std::find_if(vec.begin(), vec.end(), [rxEndTime](const auto& pair) {
        //     return pair.second == rxEndTime;
        // });
        // // get the duration of this rx
        // Time duration = phyRxIt->second - phyRxIt->first;
        // // only the last rx can use acked time
        // Time ackedTime =
        //     (std::next(phyRxIt) == vec.end()) ? packetIt->second.ackedTime : Seconds(0);

        // ChannelUtilInfo packetInfo{.duration = duration, // @todo: without PHY preamble
        //                            .txEndTime = rxEndTime,
        //                            .ackedTime = ackedTime,
        //                            .rssi = packetIt->second.rssi};
        result.push_back(ackedTime);
        // make sure AMPDU with same rx end time will be ignored
        prevEnd = ackedTime;
    }

    return result;
}

std::vector<Mac48Address>
SensingMac::GetNeighborMacAddress(uint8_t linkId) const
{
    NS_LOG_FUNCTION(this << linkId);

    std::vector<Mac48Address> stas;
    for (auto [key, value] : m_neighborStaMap)
    {
        if (value.linkId == linkId)
        {
            stas.push_back(key);
        }
    }
    return stas;
}

SensingMac::AccessInfo
SensingMac::GetLastAccessInfo(uint8_t linkId)
{
    NS_LOG_FUNCTION(this << linkId);

    AccessInfo result;
    // self info
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
