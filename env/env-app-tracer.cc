/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-app-tracer.h"

#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAppTracer");

void
EnvAppTracer::HandleAppTx(InFlightPacketMap& map,
                          uint32_t nodeId,
                          Ptr<const Packet> packet,
                          uint16_t dstPort)
{
    NS_LOG_FUNCTION(this << nodeId << packet << dstPort);

    uint64_t uid = packet->GetUid();

    if (auto it = map.find(uid); it != map.end())
    {
        NS_ABORT_MSG("Packet with UID=" << uid << " is already in map");
        return;
    }

    // Register a packet entry
    auto info = InFlightPacketInfo();
    info.m_nodeId = nodeId;
    info.m_packetSize = packet->GetSize();
    info.m_isData = true;
    info.m_dstPort = dstPort;
    info.m_appTxTime = Simulator::Now();

    map[uid] = info;
}

void
EnvAppTracer::HandleAppRx(InFlightPacketMap& map, Ptr<const Packet> packet)
{
    NS_LOG_FUNCTION(this << packet);

    uint64_t uid = packet->GetUid();

    if (auto it = map.find(uid); it == map.end())
    {
        NS_ABORT_MSG("Packet with UID=" << uid << " is not in map");
        return;
    }

    auto& info = map[uid];
    info.m_appRxTime = Simulator::Now();
}

void
EnvAppTracer::HandleBurstTx(FragmentsInBurst& burstMap,
                            Ptr<const Packet> burst,
                            const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << burst << header);

    auto burstSeq = header.GetSeq();

    if (auto it = burstMap.find(burstSeq); it != burstMap.end())
    {
        NS_ABORT_MSG("Burst with Seq=" << burstSeq << " is already in map");
        return;
    }

    // Register a burst entry
    burstMap[burstSeq] = std::vector<uint64_t>();
}

void
EnvAppTracer::HandleBurstRx(FragmentsInBurst& burstMap,
                            InFlightPacketMap& map,
                            Ptr<const Packet> burst,
                            const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << burst);

    auto burstSeq = header.GetSeq();

    if (auto it = burstMap.find(burstSeq); it == burstMap.end())
    {
        NS_ABORT_MSG("Burst with Seq=" << burstSeq << " is not in map");
        return;
    }

    for (auto fragUid : burstMap[burstSeq])
    {
        if (auto it = map.find(fragUid); it != map.end())
        {
            auto& info = map[fragUid];
            info.m_burstRx = true;
        }
        else
        {
            NS_ABORT_MSG("Fragment with UID=" << fragUid << " is not in map");
        }
    }
}

void
EnvAppTracer::HandleFragmentTx(InFlightPacketMap& map,
                               uint32_t nodeId,
                               Ptr<const Packet> fragment,
                               uint16_t dstPort,
                               const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << nodeId << fragment << dstPort);

    uint64_t uid = fragment->GetUid();

    if (auto it = map.find(uid); it != map.end())
    {
        NS_ABORT_MSG("Fragment with UID=" << uid << " is already in map");
        return;
    }

    // Register a fragment entry
    auto info = InFlightPacketInfo();
    info.m_nodeId = nodeId;
    info.m_packetSize = fragment->GetSize();
    info.m_isData = true;
    info.m_isBurst = true;
    info.m_dstPort = dstPort;
    info.m_burstSeq = header.GetSeq();
    info.m_fragSeq = header.GetFragSeq();
    info.m_appTxTime = Simulator::Now();

    map[uid] = info;
}

void
EnvAppTracer::HandleFragmentRx(InFlightPacketMap& map, Ptr<const Packet> fragment)
{
    NS_LOG_FUNCTION(this << fragment);

    uint64_t uid = fragment->GetUid();

    if (auto it = map.find(uid); it == map.end())
    {
        NS_ABORT_MSG("Fragment with UID=" << uid << " is not in map");
        return;
    }

    auto& info = map[uid];
    info.m_appRxTime = Simulator::Now();
}

} // namespace ns3