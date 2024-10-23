/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-mac-tracer.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvMacTracer");

void
EnvMacTracer::HandleEdcaEnqueue(InFlightPacketMap& map,
                                uint32_t nodeId,
                                Ptr<const WifiMpdu> item,
                                AcIndex ac)
{
    NS_LOG_FUNCTION(this << nodeId << item << ac);

    uint64_t uid = item->GetPacket()->GetUid();

    auto [it, inserted] = map.try_emplace(uid);
    if (inserted)
    {
        // packet is generated in the link layer and enqueued
        it->second = InFlightPacketInfo{.m_nodeId = nodeId, .m_packetSize = item->GetPacketSize()};
    }

    auto& info = it->second;
    info.m_srcAddress = item->GetHeader().GetAddr1();
    info.m_dstAddress = item->GetHeader().GetAddr2();
    info.m_ac = ac;
    info.m_enqueueTime = Simulator::Now();
}

void
EnvMacTracer::HandleEdcaDequeue(InFlightPacketMap& map, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << item);

    uint64_t uid = item->GetPacket()->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Packet with UID=" << uid << " is not in map");

    auto& info = map[uid];
    info.m_dequeued = true;
    info.m_dequeueTime = Simulator::Now();
}

void
EnvMacTracer::HandleEdcaExpired(InFlightPacketMap& map, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << item);

    uint64_t uid = item->GetPacket()->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Packet with UID=" << uid << " is not in map");

    auto& info = map[uid];
    info.m_expired = true;
}

void
EnvMacTracer::HandleMacForwardUp(InFlightPacketMap& map, Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);

    uint64_t uid = p->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Packet with UID=" << uid << " is not in map");

    auto& info = map[uid];
    info.m_L2RxTime = Simulator::Now();
}

void
EnvMacTracer::HandleAckedMpdu(InFlightPacketMap& map, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << item);

    uint64_t uid = item->GetPacket()->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Packet with UID=" << uid << " is not in map");

    auto& info = map[uid];
    info.m_acked = true;
}

void
EnvMacTracer::HandleCwUpdate(uint32_t nodeId, uint32_t cw, uint8_t linkId, AcIndex ac)
{
    NS_LOG_FUNCTION(this << nodeId << cw << linkId << ac);

    auto [it, inserted] = m_cw.try_emplace(nodeId, EdcaQueueMap());
    auto [acIt, inserted2] = it->second.try_emplace(ac, std::vector<std::pair<Time, uint32_t>>());
    acIt->second.emplace_back(Simulator::Now(), cw);
}

void
EnvMacTracer::HandleNewBackoff(uint32_t nodeId, uint32_t backoff, uint8_t linkId, AcIndex ac)
{
    NS_LOG_FUNCTION(this << nodeId << backoff << linkId << ac);

    auto [it, inserted] = m_backoff.try_emplace(nodeId, EdcaQueueMap());
    auto [acIt, inserted2] = it->second.try_emplace(ac, std::vector<std::pair<Time, uint32_t>>());
    acIt->second.emplace_back(Simulator::Now(), backoff);
}

} // namespace ns3