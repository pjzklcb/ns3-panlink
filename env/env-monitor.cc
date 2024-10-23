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

EnvMonitor::~EnvMonitor()
{
    NS_LOG_FUNCTION(this);
    for (auto& [nodeId, file] : m_nodeFiles)
    {
        file.close();
    }
}

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
    std::set<uint32_t> dstNodes;

    for (auto [srcNodeId, appList] : m_srcApps)
    {
        srcNodes.insert(srcNodeId);
        for (auto app : appList)
            dstNodes.insert(app.m_dst);
    }

    // Connect source app trace sources
    for (auto srcNodeId : srcNodes)
    {
        // if (appName == "udp")
        // {
        //     Config::Connect("/NodeList/" + std::to_string(srcNodeId) +
        //                         "/ApplicationList/*/$ns3::UdpClient/TxWithAddresses",
        //                     MakeCallback(&EnvMonitor::NotifyAppTx, this));
        // }

        std::string prefix = "/NodeList/" + std::to_string(srcNodeId) +
                             "/ApplicationList/*/$ns3::BurstyApplication/";
        Config::Connect(prefix + "BurstTx", MakeCallback(&EnvMonitor::NotifyBurstTx, this));
        Config::Connect(prefix + "FragmentTx", MakeCallback(&EnvMonitor::NotifyFragmentTx, this));
    }

    // Connect sink app trace sources
    for (auto dstNodeId : dstNodes)
    {
        // if (appName == "udp")
        // {
        //     Config::Connect("/NodeList/" + std::to_string(dstNodeId) +
        //                         "/ApplicationList/*/$ns3::UdpServer/RxWithAddresses",
        //                     MakeCallback(&EnvMonitor::NotifyAppRx, this));
        // }

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
    auto dstPort = InetSocketAddress::ConvertFrom(to).GetPort();

    m_appTracer->HandleAppTx(m_packetMap, nodeId, dstPort, packet);
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
    uint32_t nodeId = ContextToNodeId(context);
    auto dstPort = InetSocketAddress::ConvertFrom(to).GetPort();

    m_appTracer->HandleBurstTx(m_burstMap, nodeId, dstPort, burst, header);
}

void
EnvMonitor::NotifyBurstRx(std::string context,
                          Ptr<const Packet> burst,
                          const Address& from,
                          const Address& to,
                          const SeqTsSizeFragHeader& header)
{
    uint32_t nodeId = ContextToNodeId(context);
    auto dstPort = InetSocketAddress::ConvertFrom(to).GetPort();

    m_appTracer->HandleBurstRx(m_packetMap, m_burstMap, nodeId, dstPort, burst, header);
}

void
EnvMonitor::NotifyFragmentTx(std::string context,
                             Ptr<const Packet> fragment,
                             const Address& from,
                             const Address& to,
                             const SeqTsSizeFragHeader& header)
{
    uint32_t nodeId = ContextToNodeId(context);
    auto dstPort = InetSocketAddress::ConvertFrom(to).GetPort();

    m_appTracer->HandleFragmentTx(m_packetMap, m_burstMap, nodeId, dstPort, fragment, header);
}

void
EnvMonitor::NotifyFragmentRx(std::string context,
                             Ptr<const Packet> fragment,
                             const Address& from,
                             const Address& to,
                             const SeqTsSizeFragHeader& header)
{
    m_appTracer->HandleFragmentRx(m_packetMap, fragment, header);
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

void
EnvMonitor::SaveResultsToCsv(std::string resultDir)
{
    SaveInFlightPacketMapToCsv(resultDir);
    m_macTracer->SaveCwToCsv(resultDir);
    m_macTracer->SaveBackoffToCsv(resultDir);
}

void
EnvMonitor::SaveInFlightPacketMapToCsv(std::string resultDir)
{
    for (const auto& [uid, packet] : m_packetMap)
    {
        uint32_t nodeId = packet.m_nodeId;

        // Create a new file for this node if it doesn't exist
        if (m_nodeFiles.find(nodeId) == m_nodeFiles.end())
        {
            std::string fileName = resultDir + "/node" + std::to_string(nodeId) + "-packet.csv";
            m_nodeFiles[nodeId].open(fileName);

            // Write header line
            m_nodeFiles[nodeId]
                << "uid,size,is_data,is_burst,burst_rx,port,burst_seq,frag_seq,expired,"
                   "acked,ac,app_tx_time,enqueue_time,phy_tx_begin_time,phy_tx_end_time,"
                   "l2_rx_time,app_rx_time,tx_counter,tx_power\n";
        }

        // Write the packet to the file
        for (size_t i = 0; i < packet.m_phyTxTime.size(); i++)
        {
            m_nodeFiles[nodeId] << uid << "," << packet.m_packetSize << "," << packet.m_isData
                                << "," << packet.m_isBurst << "," << packet.m_burstRx << ","
                                << packet.m_dstPort << "," << packet.m_burstSeq << ","
                                << packet.m_fragSeq << "," << packet.m_expired << ","
                                << packet.m_acked << "," << packet.m_ac << ","
                                << packet.m_appTxTime.ToDouble(Time::MS) << ","
                                << packet.m_enqueueTime.ToDouble(Time::MS) << ","
                                << packet.m_phyTxTime[i].first.ToDouble(Time::MS) << ","
                                << packet.m_phyTxTime[i].second.ToDouble(Time::MS) << ","
                                << packet.m_L2RxTime.ToDouble(Time::MS) << ","
                                << packet.m_appRxTime.ToDouble(Time::MS) << ","
                                << packet.m_txCounter << "," << packet.m_txPowerW << std::endl;
        }
    }
}

} // namespace ns3
