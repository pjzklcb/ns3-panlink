/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-monitor.h"

#include "ns3/panlink-utils.h"
#include "ns3/ipv4-address.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvMonitor");

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
{}

void
EnvMonitor::ConnectMacTrace()
{}

void
EnvMonitor::ConnectPhyTrace()
{}

// void
// EnvMonitor::NotifyEdcaEnqueue(Ptr<const WifiMpdu> item)
// {
//     if (!item->GetHeader().IsQosData())
//     {
//         return;
//     }
//     // init a map entry if the packet's UID is not present
//     auto mapIt =
//         m_inFlightPacketMap.insert({item->GetPacket()->GetUid(), std::list<InFlightPacketInfo>()})
//             .first;

//     InFlightPacketInfo info;
//     info.m_srcAddress = item->GetHeader().GetAddr2();
//     info.m_dstAddress = item->GetHeader().GetAddr1();
//     info.m_ptrToPacket = item->GetPacket();
//     info.m_edcaEnqueueTime = Simulator::Now();

//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::NotifyAppTx(Ptr<const Packet> packet, const Address& address)
// {
//     Ptr<const Packet> p = packet;

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         return;
//     }

//     auto listIt =
//         std::find_if(mapIt->second.begin(),
//                      mapIt->second.end(),
//                      [&p](const InFlightPacketInfo& info) { return info.m_ptrToPacket == p; });

//     if (listIt == mapIt->second.end())
//     {
//         return;
//     }

//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = packet;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.m_edcaDequeueTime = listIt->m_edcaDequeueTime;
//     info.appTypeTxTime = Simulator::Now();

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW)
// {
//     totalTx += 1;
//     nodePacketTxTime[ContextToNodeId(context)][p->GetUid()].push_back(Simulator::Now());

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         return;
//     }

//     auto listIt = std::find_if(mapIt->second.begin(),
//                                mapIt->second.end(),
//                                [&p](const InFlightPacketInfo& info) {
//                                    return info.m_ptrToPacket->GetUid() == p->GetUid();
//                                });

//     if (listIt == mapIt->second.end())
//     {
//         // std::cout << "Forwarding up a packet that has not been enqueued?" << std::endl;
//         return;
//     }

//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = listIt->m_ptrToPacket;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.appTypeTxTime = listIt->appTypeTxTime;
//     info.m_phyTxTime = Simulator::Now();

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::PhyTxDoneTrace(std::string context, Ptr<const Packet> p)
// {
//     nodePacketTxEndTime[ContextToNodeId(context)][p->GetUid()].push_back(Simulator::Now());
// }

// /**
//  * PHY TX drop trace.
//  *
//  * \param context The context.
//  * \param p The packet.
//  */
// void
// EnvMonitor::PhyTxDropTrace(std::string context, Ptr<const Packet> p)
// {
//     nodePacketTxEndTime[ContextToNodeId(context)][p->GetUid()].push_back(Simulator::Now());
// }

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

// void
// EnvMonitor::NotifyAppRx(Ptr<const Packet> packet, const Address& address)
// {
//     // std::cout << "App Rx UID " << packet->GetUid() << std::endl;

//     Ptr<const Packet> p = packet;

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         // std::cout << "No packet with UID " << p->GetUid() << " is currently in queue" <<
//         // std::endl;
//         return;
//     }
//     // mapIt->second.;
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
//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = listIt->m_ptrToPacket;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.appTypeTxTime = listIt->appTypeTxTime;
//     info.m_phyTxTime = listIt->m_phyTxTime;
//     info.m_L2RxTime = listIt->m_L2RxTime;
//     info.appTypeRxTime = Simulator::Now();

//     appTxrec++;
//     // std::cout << "APRX" << std::endl;

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::NotifyMsduDequeuedFromEdcaQueue(Ptr<const WifiMpdu> item)
// {
//     // std::cout << "Dequeue UID " << item->GetPacket()->GetUid() << std::endl;

//     if (!item->GetHeader().IsQosData() || item->GetHeader().GetAddr1().IsGroup())
//     {
//         // the frame is not a unicast QoS data frame or the MSDU lifetime is higher than the
//         // max queue delay, hence the MSDU has been discarded. Do nothing in this case.
//         return;
//     }
//     Ptr<const Packet> p = item->GetPacket();

//     uint64_t srcNodeId = MacAddressToNodeId(item->GetHeader().GetAddr2());
//     auto iter = dequeueTimes.find(srcNodeId);
//     if (iter == dequeueTimes.end())
//     {
//         dequeueTimes.insert(std::make_pair(srcNodeId, Simulator::Now()));
//         return;
//     }
//     if (iter->second == Simulator::Now())
//     {
//         // std::cout << "last " << iter->second << " now " << Simulator::Now() << std::endl;
//         return;
//     }

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         // std::cout << "No packet with UID " << p->GetUid() << " is currently in queue" <<
//         // std::endl;
//         return;
//     }

//     auto listIt =
//         std::find_if(mapIt->second.begin(),
//                      mapIt->second.end(),
//                      [&p](const InFlightPacketInfo& info) { return info.m_ptrToPacket == p; });

//     if (listIt == mapIt->second.end())
//     {
//         // std::cout << "Dequeue a packet that has not been enqueued?" << std::endl;
//         return;
//     }

//     InFlightPacketInfo info;
//     info.m_srcAddress = item->GetHeader().GetAddr2();
//     info.m_dstAddress = item->GetHeader().GetAddr1();
//     info.m_ptrToPacket = item->GetPacket();
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.m_edcaDequeueTime = Simulator::Now();
//     info.appTypeTxTime = listIt->appTypeTxTime;
//     info.m_phyTxTime = listIt->m_phyTxTime;
//     info.appTypeRxTime = listIt->appTypeRxTime;
//     info.m_L2RxTime = listIt->m_L2RxTime;
//     info.m_HoLTime = std::max(listIt->m_edcaEnqueueTime, iter->second);
//     info.m_dequeued = true;

//     // HERE CALCULATE ALL DELAYS
//     double newHolSample = (info.m_edcaDequeueTime - info.m_HoLTime).ToDouble(Time::MS);
//     double queingDelay = (info.m_HoLTime - info.m_edcaEnqueueTime).ToDouble(Time::MS);
//     double accessDelay = (info.m_phyTxTime - info.m_HoLTime).ToDouble(Time::MS);
//     double txDelay = (info.m_edcaDequeueTime - info.m_phyTxTime).ToDouble(Time::MS);

//     auto it = edcaHolSample.find(srcNodeId);
//     auto ite = intervalEdcaHolSample.find(srcNodeId);
//     if (ite == intervalEdcaHolSample.end())
//     {
//         std::vector<double> tiVec;
//         tiVec.push_back(newHolSample);
//         tiVec.push_back(queingDelay);
//         tiVec.push_back(accessDelay);
//         tiVec.push_back(txDelay);
//         intervalEdcaHolSample.insert(std::make_pair(srcNodeId, tiVec));
//         // std::cout << "first Sample: " << newHolSample << "\n";
//     }
//     else
//     {
//         ite->second.push_back(newHolSample);
//         ite->second.push_back(queingDelay);
//         ite->second.push_back(accessDelay);
//         ite->second.push_back(txDelay);
//     }
//     if (it == edcaHolSample.end())
//     {
//         std::vector<double> tiVec;
//         tiVec.push_back(newHolSample);
//         tiVec.push_back(queingDelay);
//         tiVec.push_back(accessDelay);
//         tiVec.push_back(txDelay);
//         edcaHolSample.insert(std::make_pair(srcNodeId, tiVec));
//         // std::cout << "first Sample: " << newHolSample << "\n";
//     }
//     else
//     {
//         it->second.push_back(newHolSample);
//         it->second.push_back(queingDelay);
//         it->second.push_back(accessDelay);
//         it->second.push_back(txDelay);
//     }

//     iter->second = Simulator::Now();
//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::NotifyMsduDroppedAfterDequeueFromEdcaQueue(Ptr<const WifiMpdu> item)
// {
//     // std::cout << "did it work?" << std::endl;
// }

// void
// EnvMonitor::FragmentTx(Ptr<const Packet> fragment,
//                                 const Address& from,
//                                 const Address& to,
//                                 const SeqTsSizeFragHeader& header)
// {
//     NS_LOG_INFO("Sent fragment " << header.GetFragSeq() << "/" << header.GetFrags()
//                                  << " of burst seq=" << header.GetSeq()
//                                  << " of header.GetSize ()=" << header.GetSize()
//                                  << " (fragment->GetSize ()=" << fragment->GetSize()
//                                  << ") bytes from " << AddressToString(from) << " to "
//                                  << AddressToString(to) << " at " << header.GetTs().As(Time::S));
//     Ptr<const Packet> p = fragment;

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         // std::cout << "No packet with UID " << p->GetUid() << " is currently in queue" <<
//         // std::endl;
//         return;
//     }

//     auto listIt =
//         std::find_if(mapIt->second.begin(),
//                      mapIt->second.end(),
//                      [&p](const InFlightPacketInfo& info) { return info.m_ptrToPacket == p; });

//     if (listIt == mapIt->second.end())
//     {
//         // std::cout << "Forwarding up a packet that has not been enqueued?" << std::endl;
//         return;
//     }

//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = listIt->m_ptrToPacket;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.m_edcaDequeueTime = listIt->m_edcaDequeueTime;
//     info.appTypeTxTime = Simulator::Now();

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
// }

// void
// EnvMonitor::FragmentRx(Ptr<const Packet> fragment,
//                                 const Address& from,
//                                 const Address& to,
//                                 const SeqTsSizeFragHeader& header)
// {
//     NS_LOG_INFO("Received fragment "
//                 << header.GetFragSeq() << "/" << header.GetFrags() << " of burst seq="
//                 << header.GetSeq() << " of header.GetSize ()=" << header.GetSize()
//                 << " (fragment->GetSize ()=" << fragment->GetSize() << ") bytes from "
//                 << AddressToString(from) << " to " << AddressToString(to) << " at "
//                 << header.GetTs().As(Time::S));
//     Ptr<const Packet> p = fragment;

//     auto mapIt = m_inFlightPacketMap.find(p->GetUid());
//     if (mapIt == m_inFlightPacketMap.end())
//     {
//         std::cout << "No packet with UID " << p->GetUid() << " is currently in queue" << std::endl;
//         return;
//     }
//     // mapIt->second.;
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
//     InFlightPacketInfo info;
//     info.m_srcAddress = listIt->m_srcAddress;
//     info.m_dstAddress = listIt->m_dstAddress;
//     info.m_ptrToPacket = listIt->m_ptrToPacket;
//     info.m_edcaEnqueueTime = listIt->m_edcaEnqueueTime;
//     info.appTypeTxTime = listIt->appTypeTxTime;
//     info.m_phyTxTime = listIt->m_phyTxTime;
//     info.m_L2RxTime = listIt->m_L2RxTime;
//     info.appTypeRxTime = Simulator::Now();
//     // if (p->GetSize() >= pktSize) // ignore non-data frames
//     // {
//     appTxrec++;
//     // }
//     // else
//     // {
//     //     std::cout << "Size " << p->GetSize() << std::endl;
//     // }
//     // std::cout << "APRX" << std::endl;

//     mapIt->second.erase(listIt);
//     mapIt->second.insert(mapIt->second.end(), info);
//     // appTxrec++;
// }

// void
// EnvMonitor::BurstTx(Ptr<const Packet> burst,
//                             const Address& from,
//                             const Address& to,
//                             const SeqTsSizeFragHeader& header)
// {
//     NS_LOG_INFO("Sent burst seq=" << header.GetSeq() << " of header.GetSize ()=" << header.GetSize()
//                                   << " (burst->GetSize ()=" << burst->GetSize() << ") bytes from "
//                                   << AddressToString(from) << " to " << AddressToString(to)
//                                   << " at " << header.GetTs().As(Time::S));
//     // std::cout << "APPTX" << std::endl;
// }

// void
// EnvMonitor::BurstRx(Ptr<const Packet> burst,
//                             const Address& from,
//                             const Address& to,
//                             const SeqTsSizeFragHeader& header)
// {
//     NS_LOG_INFO("Received burst seq="
//                 << header.GetSeq() << " of header.GetSize ()=" << header.GetSize()
//                 << " (burst->GetSize ()=" << burst->GetSize() << ") bytes from "
//                 << AddressToString(from) << " to " << AddressToString(to) << " at "
//                 << header.GetTs().As(Time::S));
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
