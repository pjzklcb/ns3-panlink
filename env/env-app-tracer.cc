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
                          uint16_t dstPort,
                          Ptr<const Packet> packet)
{
    NS_LOG_FUNCTION(this << nodeId << dstPort << packet);

    uint64_t uid = packet->GetUid();

    // Register an entry in the packet map
    auto [packetIt, inserted] = map.try_emplace(uid);
    NS_ABORT_MSG_IF(!inserted, "Packet with UID=" << uid << " already exists");

    packetIt->second = InFlightPacketInfo{.m_nodeId = nodeId,
                                          .m_packetSize = packet->GetSize(),
                                          .m_isData = true,
                                          .m_dstPort = dstPort,
                                          .m_appTxTime = Simulator::Now()};
}

void
EnvAppTracer::HandleAppRx(InFlightPacketMap& map, Ptr<const Packet> packet)
{
    NS_LOG_FUNCTION(this << packet);

    uint64_t uid = packet->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Packet with UID=" << uid << " does not exist.");

    map[uid].m_appRxTime = Simulator::Now();
}

void
EnvAppTracer::HandleBurstTx(BurstMap& burstMap,
                            uint32_t nodeId,
                            uint16_t dstPort,
                            Ptr<const Packet> burst,
                            const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << nodeId << dstPort << burst);

    uint32_t seqN = header.GetSeq();

    // Register an entry in the burst map
    auto [nodeIt, nodeInserted] = burstMap.try_emplace(nodeId);
    auto [portIt, portInserted] = nodeIt->second.try_emplace(dstPort);
    auto [seqIt, inserted] = portIt->second.try_emplace(seqN);
    NS_ABORT_MSG_IF(!inserted && seqIt->second.size() > 0,
                    "Node-" << nodeId << ", port=" << dstPort << ": Burst with seqN=" << seqN
                            << " already exists");
}

void
EnvAppTracer::HandleBurstRx(InFlightPacketMap& map,
                            BurstMap& burstMap,
                            uint32_t nodeId,
                            uint16_t dstPort,
                            Ptr<const Packet> burst,
                            const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << nodeId << dstPort << burst << header);

    uint32_t seqN = header.GetSeq();

    for (auto fragUid : burstMap[nodeId][dstPort][seqN])
    {
        NS_ABORT_MSG_IF(map.find(fragUid) == map.end(),
                        "Fragment with UID=" << fragUid << " does not exist.");
        map[fragUid].m_burstRx = true;
    }
}

void
EnvAppTracer::HandleFragmentTx(InFlightPacketMap& map,
                               BurstMap& burstMap,
                               uint32_t nodeId,
                               uint16_t dstPort,
                               Ptr<const Packet> fragment,
                               const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << nodeId << dstPort << fragment << header);

    uint64_t uid = fragment->GetUid();

    // Register an entry in the packet map
    auto [packetIt, inserted] = map.try_emplace(uid);
    NS_ABORT_MSG_IF(!inserted, "Fragment with UID=" << uid << " already exists.");

    packetIt->second = InFlightPacketInfo{.m_nodeId = nodeId,
                                          .m_packetSize = fragment->GetSize(),
                                          .m_isData = true,
                                          .m_isBurst = true,
                                          .m_dstPort = dstPort,
                                          .m_burstSeq = header.GetSeq(),
                                          .m_fragSeq = header.GetFragSeq(),
                                          .m_appTxTime = Simulator::Now()};

    // Create an entry in the burst map
    uint64_t seqN = header.GetSeq();
    burstMap[nodeId][dstPort][seqN].emplace_back(uid);
}

void
EnvAppTracer::HandleFragmentRx(InFlightPacketMap& map,
                               Ptr<const Packet> fragment,
                               const SeqTsSizeFragHeader& header)
{
    NS_LOG_FUNCTION(this << fragment);

    uint64_t uid = fragment->GetUid();
    NS_ABORT_MSG_IF(map.find(uid) == map.end(), "Fragment with UID=" << uid << " does not exist.");

    // Find the fragment
    map[uid].m_appRxTime = Simulator::Now();
}

} // namespace ns3