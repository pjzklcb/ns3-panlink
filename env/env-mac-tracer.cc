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
EnvMacTracer::HandleEdcaDequeue(InFlightPacketMap& map,
                                uint32_t nodeId,
                                Ptr<const WifiMpdu> item,
                                AcIndex ac)
{
    NS_LOG_FUNCTION(this << item);

    uint64_t uid = item->GetPacket()->GetUid();
    if (map.find(uid) == map.end())
    {
        // some mgt packets like probe request are not enqueued
        NS_LOG_DEBUG("Packet with UID=" << uid << " dequeued without enqueuing");
        map[uid] = InFlightPacketInfo{.m_nodeId = nodeId,
                                      .m_packetSize = item->GetPacketSize(),
                                      .m_srcAddress = item->GetHeader().GetAddr1(),
                                      .m_dstAddress = item->GetHeader().GetAddr2(),
                                      .m_ac = ac};
    }

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

    auto [it, inserted] = m_cw.try_emplace(nodeId);
    auto [acIt, inserted2] = it->second.try_emplace(ac);
    acIt->second[Simulator::Now()] = {cw};
}

void
EnvMacTracer::HandleNewBackoff(uint32_t nodeId, uint32_t backoff, uint8_t linkId, AcIndex ac)
{
    NS_LOG_FUNCTION(this << nodeId << backoff << linkId << ac);

    auto [it, inserted] = m_backoff.try_emplace(nodeId);
    auto [acIt, inserted2] = it->second.try_emplace(ac);
    acIt->second[Simulator::Now()] = {backoff};
}

void
EnvMacTracer::HandleWaitedSlotsUpdate(uint32_t nodeId,
                                      uint32_t nDeferredAccess,
                                      uint32_t waitedSlots,
                                      uint32_t backoff,
                                      uint8_t linkId,
                                      AcIndex ac)
{
    NS_LOG_FUNCTION(this << nodeId << nDeferredAccess << waitedSlots << backoff << linkId << ac);

    auto [it, inserted] = m_waitedSlots.try_emplace(nodeId);
    auto [acIt, inserted2] = it->second.try_emplace(ac);
    acIt->second[Simulator::Now()] = {nDeferredAccess, waitedSlots, backoff};
}

void
EnvMacTracer::WriteCwToCsv(std::string resultDir)
{
    std::ofstream os(resultDir + "/cw.csv");
    os << "node_id,ac,time,cw\n";
    for (auto& [nodeId, acMap] : m_cw)
    {
        for (auto& [ac, timeMap] : acMap)
        {
            for (auto& [time, cw] : timeMap)
            {
                os << nodeId << "," << ac << "," << time.ToDouble(Time::MS) << "," << cw[0] << "\n";
            }
        }
    }
}

void
EnvMacTracer::WriteBackoffToCsv(std::string resultDir)
{
    std::ofstream os(resultDir + "/backoff.csv");
    os << "node_id,ac,time,backoff\n";
    for (auto& [nodeId, acMap] : m_backoff)
    {
        for (auto& [ac, timeMap] : acMap)
        {
            for (auto& [time, backoff] : timeMap)
            {
                os << nodeId << "," << ac << "," << time.ToDouble(Time::MS) << "," << backoff[0]
                   << "\n";
            }
        }
    }
}

void
EnvMacTracer::WriteWaitedSlotsToCsv(std::string resultDir)
{
    std::ofstream os(resultDir + "/waited_slots.csv");
    os << "node_id,ac,time,n_deferred,waited_slots,backoff\n";
    for (auto& [nodeId, acMap] : m_waitedSlots)
    {
        for (auto& [ac, timeMap] : acMap)
        {
            for (auto& [time, params] : timeMap)
            {
                os << nodeId << "," << ac << "," << time.ToDouble(Time::MS) << "," << params[0]
                   << "," << params[1] << "," << params[2] << "\n";
            }
        }
    }
}

} // namespace ns3