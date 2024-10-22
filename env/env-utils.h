/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PACKET_INFO_H
#define PACKET_INFO_H

#include "ns3/mac48-address.h"
#include "ns3/nstime.h"
#include "ns3/qos-utils.h"

namespace ns3
{

/**
 * Types of AI agents.
 */
enum AgentType
{
    NO_RL = 0,
    RL_CCA_SENSITIVITY,
    RL_CHANNEL_ACCESS,
    RL_RATE_CONTROL,
    RL_AGGREGATION
};

/**
 * Record all the in-flight packet information
 */
struct InFlightPacketInfo
{
    /* APP */
    uint32_t m_nodeId; // source node ID
    uint32_t m_packetSize;
    bool m_isData{false};
    bool m_isBurst{false};
    bool m_burstRx{false}; // the burst is successfully received
    uint32_t m_dstPort{0};
    uint32_t m_burstSeq{0};
    uint32_t m_fragSeq{0};
    Time m_appTxTime{Seconds(0)}; // time the packet was created by app
    Time m_appRxTime{Seconds(0)}; // time the packet was received by app

    /* MAC */
    bool m_expired{false};
    bool m_dequeued{false};
    bool m_acked{false};
    Mac48Address m_srcAddress;
    Mac48Address m_dstAddress;
    AcIndex m_ac{AC_BE_NQOS};
    Time m_enqueueTime{Seconds(0)}; // time the packet was enqueued into an EDCA queue
    Time m_dequeueTime{Seconds(0)}; // time the packet was dequeued from EDCA queue
    Time m_HoLTime{Seconds(0)};     // time the packet became Head of Line
    Time m_L2RxTime{Seconds(0)};    // time packet got forwarded up by the Mac Layer (L2 latency =
                                    // L2RxTime - EnqueueTime)

    /* PHY */
    uint32_t m_txCounter{0}; // transmission counter
    std::vector<std::pair<Time, Time>> m_phyTxTime{
        {{Seconds(0), Seconds(0)}}}; // time the packet was sent by PHY
    double m_txPowerW;
};

typedef std::map<uint64_t /* UID */, InFlightPacketInfo> InFlightPacketMap;
typedef std::map<uint32_t /* seqN */, std::vector<uint64_t /* UID */>> FragmentsInBurst;

/**
 * Parse context strings of the form "/NodeList/x/DeviceList/x/..." to extract the node ID.
 *
 * \param context The context to parse.
 * \return the NodeId
 */
uint32_t ContextToNodeId(std::string context);

/**
 * Get the node ID of the specified MAC address
 *
 * \param map the address map
 * \param address the specified MAC address
 * \return the node ID
 */
uint32_t MacAddressToNodeId(std::map<Mac48Address, uint32_t>& map, Mac48Address address);

/**
 * Get the AC index of the specified string
 *
 * \param ac the string
 * \return the AC index
 */
AcIndex StringToAcIndex(std::string ac);

/**
 * Get the string of the specified AC index
 *
 * \param ac the AC index
 * \return the string
 */
std::string AcIndexToString(AcIndex ac);

/**
 * Parse context strings of the form ".../$ns3::WifiMac/BE_Txop/..." to extract the AC index.
 *
 * \param context the context
 * \return the AC index
 */
AcIndex ContextToAcIndex(std::string context);

} // namespace ns3

#endif /* PACKET_INFO_H */
