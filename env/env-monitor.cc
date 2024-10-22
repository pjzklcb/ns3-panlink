/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-monitor.h"

#include "ns3/config.h"
#include "ns3/ipv4-address.h"
#include "ns3/panlink-utils.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvMonitor");

void
EnvMonitor::SetNodes(NodeContainer nodes)
{
    m_nodes = nodes;
}

void
EnvMonitor::SetDevices(NetDeviceContainer devices)
{
    m_devices = devices;
}

void
EnvMonitor::SetIpv4Interfaces(Ipv4InterfaceContainer ipv4Interfaces)
{
    m_ipv4Interfaces = ipv4Interfaces;
}

void
EnvMonitor::SetSrcApps(SrcAppMap srcApps)
{
    m_srcApps = srcApps;
}

void
EnvMonitor::ConnectTraceSources()
{
    NS_LOG_FUNCTION(this);

    m_appTracer = CreateObject<EnvAppTracer>();
    m_macTracer = CreateObject<EnvMacTracer>();
    m_phyTracer = CreateObject<EnvPhyTracer>();

    ConnectAppTrace();
    ConnectMacTrace();
    ConnectPhyTrace();
    Simulator::ScheduleNow(&EnvMonitor::ConnectEdcaQueueTrace, this);
}

void
EnvMonitor::ConnectAppTrace()
{
    std::set<uint32_t> srcNodes;
    std::set<uint32_t> snkNodes;

    for (auto [srcNodeId, appList] : m_srcApps)
    {
        srcNodes.insert(srcNodeId);
        for (auto app : appList)
            snkNodes.insert(app.m_dst);
    }

    // Connect source app trace sources
    for (auto srcNodeId : srcNodes)
    {
        std::string prefix = "/NodeList/" + std::to_string(srcNodeId) +
                             "/ApplicationList/*/$ns3::BurstyApplication/";
        Config::Connect(prefix + "BurstTx", MakeCallback(&EnvMonitor::NotifyBurstTx, this));
        Config::Connect(prefix + "FragmentTx", MakeCallback(&EnvMonitor::NotifyFragmentTx, this));
    }

    // Connect sink app trace sources
    for (auto dstNodeId : snkNodes)
    {
        std::string prefix =
            "/NodeList/" + std::to_string(dstNodeId) + "/ApplicationList/*/$ns3::BurstSink/";
        Config::Connect(prefix + "BurstRx", MakeCallback(&EnvMonitor::NotifyBurstRx, this));
        Config::Connect(prefix + "FragmentRx", MakeCallback(&EnvMonitor::NotifyFragmentRx, this));
    }
}

void
EnvMonitor::ConnectMacTrace()
{
    // Trace MAC forward up
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/MacRx",
                    MakeCallback(&EnvMonitor::NotifyMacForwardUp, this));

    // Trace MPDU ack
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/AckedMpdu",
                    MakeCallback(&EnvMonitor::NotifyAckedMpdu, this));

    for (const auto& ac : {"BE", "BK", "VI", "VO"})
    {
        std::string prefix = "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::WifiMac/";

        // Trace CW update
        Config::Connect(prefix + ac + "_Txop/CwTrace",
                        MakeCallback(&EnvMonitor::NotifyCwUpdate, this));

        // Trace new backoff
        Config::Connect(prefix + ac + "_Txop/BackoffTrace",
                        MakeCallback(&EnvMonitor::NotifyNewBackoff, this));
    }
}

void
EnvMonitor::ConnectEdcaQueueTrace()
{
    for (auto devIt = m_devices.Begin(); devIt != m_devices.End(); ++devIt)
    {
        auto mac = DynamicCast<WifiNetDevice>(*devIt)->GetMac();
        for (const auto& ac : {AC_BE, AC_BK, AC_VI, AC_VO})
        {
            std::string name = AcIndexToString(ac);
            mac->GetTxopQueue(ac)->TraceConnect("Enqueue",
                                                name,
                                                MakeCallback(&EnvMonitor::NotifyEdcaEnqueue, this));
            mac->GetTxopQueue(ac)->TraceConnectWithoutContext(
                "Dequeue",
                MakeCallback(&EnvMonitor::NotifyEdcaDequeue, this));
            mac->GetTxopQueue(ac)->TraceConnectWithoutContext(
                "Expired",
                MakeCallback(&EnvMonitor::NotifyEdcaExpired, this));
        }
    }
}

void
EnvMonitor::ConnectPhyTrace()
{
    // Trace PHY TX begin
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxBegin",
                    MakeCallback(&EnvMonitor::NotifyPhyTxBegin, this));

    // Trace PHY TX end
    Config::Connect("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/PhyTxEnd",
                    MakeCallback(&EnvMonitor::NotifyPhyTxEnd, this));
}

void
EnvMonitor::NotifyAppTx(std::string context,
                        Ptr<const Packet> packet,
                        const Address& from,
                        const Address& to)
{
    uint32_t nodeId = ContextToNodeId(context);
    auto inet = InetSocketAddress::ConvertFrom(to);

    m_appTracer->HandleAppTx(m_packetMap, nodeId, packet, inet.GetPort());
}

void
EnvMonitor::NotifyAppRx(std::string context,
                        Ptr<const Packet> packet,
                        const Address& from,
                        const Address& to)
{
    m_appTracer->HandleAppRx(m_packetMap, packet);
}

void
EnvMonitor::NotifyBurstTx(std::string context,
                          Ptr<const Packet> burst,
                          const Address& from,
                          const Address& to,
                          const SeqTsSizeFragHeader& header)
{
    m_appTracer->HandleBurstTx(m_burstMap, burst, header);
}

void
EnvMonitor::NotifyBurstRx(std::string context,
                          Ptr<const Packet> burst,
                          const Address& from,
                          const Address& to,
                          const SeqTsSizeFragHeader& header)
{
    m_appTracer->HandleBurstRx(m_burstMap, m_packetMap, burst, header);
}

void
EnvMonitor::NotifyFragmentTx(std::string context,
                             Ptr<const Packet> fragment,
                             const Address& from,
                             const Address& to,
                             const SeqTsSizeFragHeader& header)
{
    uint32_t nodeId = ContextToNodeId(context);
    auto inet = InetSocketAddress::ConvertFrom(to);

    m_appTracer->HandleFragmentTx(m_packetMap, nodeId, fragment, inet.GetPort(), header);
}

void
EnvMonitor::NotifyFragmentRx(std::string context,
                             Ptr<const Packet> fragment,
                             const Address& from,
                             const Address& to,
                             const SeqTsSizeFragHeader& header)
{
    m_appTracer->HandleFragmentRx(m_packetMap, fragment);
}

void
EnvMonitor::NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item)
{
    if (!item->GetHeader().IsQosData())
    {
        return;
    }

    auto nodeId = MacAddressToNodeId(m_macAddressMap, item->GetHeader().GetAddr2());

    m_macTracer->HandleEdcaEnqueue(m_packetMap, nodeId, item, StringToAcIndex(ac));
}

void
EnvMonitor::NotifyEdcaDequeue(Ptr<const WifiMpdu> item)
{
    m_macTracer->HandleEdcaDequeue(m_packetMap, item);
}

void
EnvMonitor::NotifyEdcaExpired(Ptr<const WifiMpdu> item)
{
    m_macTracer->HandleEdcaExpired(m_packetMap, item);
}

void
EnvMonitor::NotifyMacForwardUp(std::string context, Ptr<const Packet> p)
{
    m_macTracer->HandleMacForwardUp(m_packetMap, p);
}

void
EnvMonitor::NotifyAckedMpdu(std::string context, Ptr<const WifiMpdu> item)
{
    m_macTracer->HandleAckedMpdu(m_packetMap, item);
}

void
EnvMonitor::NotifyCwUpdate(std::string context, uint32_t cw, uint8_t linkId)
{
    uint32_t nodeId = ContextToNodeId(context);
    AcIndex ac = ContextToAcIndex(context);
    m_macTracer->HandleCwUpdate(nodeId, cw, linkId, ac);
}

void
EnvMonitor::NotifyNewBackoff(std::string context, uint32_t backoff, uint8_t linkId)
{
    uint32_t nodeId = ContextToNodeId(context);
    AcIndex ac = ContextToAcIndex(context);
    m_macTracer->HandleNewBackoff(nodeId, backoff, linkId, ac);
}

void
EnvMonitor::NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW)
{
    uint32_t nodeId = ContextToNodeId(context);

    m_phyTracer->HandlePhyTxBegin(m_packetMap, nodeId, p, txPowerW);
}

void
EnvMonitor::NotifyPhyTxEnd(std::string context, Ptr<const Packet> p)
{
    m_phyTracer->HandlePhyTxEnd(m_packetMap, p);
}

} // namespace ns3
