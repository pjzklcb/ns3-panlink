/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "sensing-mac.h"

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
                                          TimeValue(Seconds(0.05)),
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
SensingMac::MacTx(Ptr<const Packet> packet)
{
    NS_LOG_FUNCTION(this << packet);
}

void
SensingMac::MacRx(Ptr<const Packet> packet)
{
    NS_LOG_FUNCTION(this << packet);
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

void
SensingMac::AckedMpdu(Ptr<const WifiMpdu> mpdu)
{
}

} // namespace ns3
