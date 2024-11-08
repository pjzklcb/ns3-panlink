/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PACKET_INFO_H
#define PACKET_INFO_H

#include "ns3/core-module.h"
#include "ns3/wifi-module.h"


namespace ns3
{

/**
 * Record all the in-flight packet information
 */
struct PacketInfo
{
    /* APP */
    bool m_isData{false};
    uint32_t m_port{0};
    Time m_appTxTime{Seconds(0)};     // time the packet was created by app (E2E)
    Time m_appRxTime{Seconds(0)};     // time the packet was received by app (E2E)

    Mac48Address m_srcAddress;
    Mac48Address m_dstAddress;
    AcIndex m_ac;
    uint32_t m_packetSize;
    Time m_edcaEnqueueTime{Seconds(0)}; // time the packet was enqueued into an EDCA queue
    Time m_edcaDequeueTime{Seconds(0)}; // time the packet was dequeued from EDCA queue
    
    Time m_HoLTime{Seconds(0)};         // time the packet became Head of Line
    Time m_L2RxTime{Seconds(0)};  // time packet got forwarded up to the Mac Layer (L2 latency =
                                  // L2RxTime - edcaEnqueueTime)
    Time m_phyTxTime{Seconds(0)}; // time packet began transmission
    bool m_dequeued{false};
};

} // namespace ns3

#endif /* PACKET_INFO_H */
