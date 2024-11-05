/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "sensing-mac.h"

#include "ns3/wifi-mpdu.h"
#include "ns3/env-utils.h"

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
                                          MakeTimeAccessor(&SensingMac::SetDuration),
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
SensingMac::DoDispose()
{
    NS_LOG_FUNCTION(this);
    m_mac = nullptr;
}

void
SensingMac::SetWifiMac(const Ptr<WifiMac> mac)
{
    NS_LOG_FUNCTION(this << mac);
    m_mac = mac;
}

void
SensingMac::SetDuration(Time duration)
{
    NS_LOG_FUNCTION(this << duration);
    m_duration = duration;
}

void
SensingMac::NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item)
{
    NS_LOG_FUNCTION(this << ac << item);

    auto uid = item->GetPacket()->GetUid();
    auto [it, inserted] = m_selfPacketMap.try_emplace(uid);
    NS_ABORT_MSG_IF(!inserted, "Packet with uid=" << uid << " already exists");

    it->second.m_uid = uid;
    it->second.m_packetSize = item->GetPacketSize();
    it->second.m_ac = StringToAcIndex(ac);
    it->second.m_enqueueTime = Simulator::Now();
    it->second.m_duration = item->GetHeader().GetDuration();
}

void
SensingMac::NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW)
{
    NS_LOG_FUNCTION(this << context << p << txPowerW);

    auto linkId = static_cast<uint8_t>(std::stoi(context));
    auto it = m_selfPacketMap.find(p->GetUid());
    NS_ABORT_MSG_IF(it == m_selfPacketMap.cend(), "Packet with uid=" << p->GetUid() << " not found");

    it->second.m_linkId = linkId;
    it->second.m_phyTxTime.push_back(Simulator::Now());
    it->second.m_txPowerW = txPowerW;
}

void
SensingMac::AckedMpdu(Ptr<const WifiMpdu> mpdu)
{
    NS_LOG_FUNCTION(this << mpdu);

    auto uid = mpdu->GetPacket()->GetUid();
    auto it = m_selfPacketMap.find(uid);
    NS_ABORT_MSG_IF(it == m_selfPacketMap.cend(), "Packet with uid=" << uid << " not found");

    it->second.m_ackedTime = Simulator::Now();
}

void 
SensingMac::WipeAllExpiredPackets()
{
    NS_LOG_FUNCTION(this);

    for (auto it = m_selfPacketMap.begin(); it != m_selfPacketMap.end();)
    {
        if (Simulator::Now() - it->second.m_enqueueTime > m_duration)
        {
            it = m_selfPacketMap.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

void
SensingMac::FrameExchangeSensing(Mac48Address sender, double rssi, Time duration, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << sender << rssi << duration);
}

void
SensingMac::WifiMacSensing(Ptr<const WifiMpdu> mpdu, uint8_t linkId)
{
    NS_LOG_FUNCTION(this << mpdu << linkId);
}

} // namespace ns3
