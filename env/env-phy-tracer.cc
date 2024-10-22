/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-phy-tracer.h"

#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvPhyTracer");

void
EnvPhyTracer::HandlePhyTxBegin(InFlightPacketMap& map,
                               uint32_t nodeId,
                               Ptr<const Packet> p,
                               double txPowerW)
{
    NS_LOG_FUNCTION(this << p << txPowerW);

    uint64_t uid = p->GetUid();

    auto [it, inserted] = map.try_emplace(uid, InFlightPacketInfo());
    if (inserted)
    {
        // packet is not EDCA enqueued
        it->second.m_nodeId = nodeId;
        it->second.m_packetSize = p->GetSize();
    }

    auto& info = it->second;
    ++info.m_txCounter;
    if (info.m_txCounter == 1)
    {
        info.m_phyTxTime[0].first = Simulator::Now();
    }
    else
    {
        info.m_phyTxTime.push_back({Simulator::Now(), Seconds(0)});
    }
    info.m_txPowerW = txPowerW;

    // Record air interface occupation time
    auto& airVec = m_airTime[nodeId];
    airVec.push_back({Simulator::Now(), Seconds(0)});
}

void 
EnvPhyTracer::HandlePhyTxEnd(InFlightPacketMap& map, Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p);

    uint64_t uid = p->GetUid();

    auto it = map.find(uid);
    if (it == map.end())
    {
        NS_ABORT_MSG("Packet with UID " << uid << " is not in map");
        return;
    }

    auto& info = it->second;
    info.m_phyTxTime.back().second = Simulator::Now();

    // Record air interface occupation time
    auto& airVec = m_airTime[info.m_nodeId];
    airVec.back().second = Simulator::Now();
}

} // namespace ns3