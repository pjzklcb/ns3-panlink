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
}

void
EnvMonitor::ConnectAppTrace()
{
    std::set<std::pair<uint32_t, std::string>> srcAppList;
    std::set<std::pair<uint32_t, std::string>> snkAppList;

    for (auto [srcNodeId, appList] : m_srcApps)
    {
        for (auto app : appList)
        {
            srcAppList.insert(std::make_pair(srcNodeId, app.m_name));
            snkAppList.insert(std::make_pair(app.m_dst, app.m_name));
        }
    }

    // Connect source app trace sources
    for (auto [srcNodeId, appName] : srcAppList)
    {
        if (appName == "udp")
        {
            Config::Connect("/NodeList/" + std::to_string(srcNodeId) +
                                "/ApplicationList/*/$ns3::UdpClient/TxWithAddresses",
                            MakeCallback(&EnvMonitor::NotifyAppTx, this));
        }
        else
        {
            std::string configPath = "/NodeList/" + std::to_string(srcNodeId) +
                                     "/ApplicationList/*/$ns3::BurstyApplication/";
            Config::Connect(configPath + "BurstTx", MakeCallback(&EnvMonitor::NotifyBurstTx, this));
            Config::Connect(configPath + "FragmentTx",
                            MakeCallback(&EnvMonitor::NotifyFragmentTx, this));
        }
    }

    // Connect sink app trace sources
    for (auto [dstNodeId, appName] : snkAppList)
    {
        if (appName == "udp")
        {
            Config::Connect("/NodeList/" + std::to_string(dstNodeId) +
                                "/ApplicationList/*/$ns3::UdpServer/RxWithAddresses",
                            MakeCallback(&EnvMonitor::NotifyAppRx, this));
        }
        else
        {
            std::string configPath =
                "/NodeList/" + std::to_string(dstNodeId) + "/ApplicationList/*/$ns3::BurstSink/";
            Config::Connect(configPath + "BurstRx", MakeCallback(&EnvMonitor::NotifyBurstRx, this));
            Config::Connect(configPath + "FragmentRx",
                            MakeCallback(&EnvMonitor::NotifyFragmentRx, this));
        }
    }
}

void
EnvMonitor::ConnectMacTrace()
{
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

// void
// EnvMonitor::NotifyEdcaEnqueue(Ptr<const WifiMpdu> item)
// {
//     if (!item->GetHeader().IsQosData())
//     {
//         return;
//     }
//     // init a map entry if the packet's UID is not present
//     auto mapIt =
//         m_inFlightPacketMap.insert({item->GetPacket()->GetUid(),
//         std::list<InFlightPacketInfo>()})
//             .first;

//     InFlightPacketInfo info;
//     info.m_srcAddress = item->GetHeader().GetAddr2();
//     info.m_dstAddress = item->GetHeader().GetAddr1();
//     info.m_ptrToPacket = item->GetPacket();
//     info.m_edcaEnqueueTime = Simulator::Now();

//     mapIt->second.insert(mapIt->second.end(), info);
// }

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

// void
// EnvMonitor::NotifyMacForwardUp(Ptr<const Packet> p)
// {
//     // std::cout << "MacForwardUp UID " << p->GetUid() << std::endl;
//     // Ptr<const Packet> p = packet;

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         // std::cout << "No packet with UID " << p->GetUid() << " is currently in queue" <<
//         // std::endl;
//         return;
//     }

//     auto listIt = std::find_if(mapIt->second.begin(),
//                                mapIt->second.end(),
//                                [&p](const InFlightPacketInfo& info) {
//                                    return info.m_ptrToPacket->GetUid() == p->GetUid();
//                                });

//     if (listIt == mapIt->second.end())
//     {
//         std::cout << "Forwarding up a packet that has not been enqueued?" << std::endl;
//         return;
//     }
//     if (listIt->m_dstAddress.IsGroup())
//     {
//         return;
//     }

//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = listIt->m_ptrToPacket;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.appTypeTxTime = listIt->appTypeTxTime;
//     info.m_phyTxTime = listIt->m_phyTxTime;
//     // info.appTypeRxTime = Simulator::Now();
//     info.m_L2RxTime = Simulator::Now();

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// /**
//  * Contention window trace.
//  *
//  * \param context The context.
//  * \param cw The contention window.
//  */
// void
// EnvMonitor::CwTrace(std::string context, uint32_t cw, uint8_t /* linkId */)
// {
//     // std::cout << Simulator::Now().GetSeconds() << " " << ContextToNodeId(context) << " " <<
//     // cw
//     //           << std::endl;
//     nodeCw[ContextToNodeId(context)].push_back(cw);
// }

// /**
//  * Backoff trace.
//  *
//  * \param context The context.
//  * \param newVal The backoff value.
//  */
// void
// EnvMonitor::BackoffTrace(std::string context, uint32_t newVal, uint8_t /* linkId */)
// {
//     // std::cout << Simulator::Now().GetSeconds() << " " << ContextToNodeId(context) << " " <<
//     // newVal
//     //           << std::endl;
//     nodeBackoff[ContextToNodeId(context)].push_back(newVal);
// }

} // namespace ns3
