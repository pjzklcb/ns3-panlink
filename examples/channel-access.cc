/*
 * Copyright (c) 2024
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/enum.h"
#include "ns3/he-phy.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/on-off-helper.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/packet-sink.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/udp-server.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-acknowledgment.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"

#include <functional>

// This is a simple example in order to show how to configure an PanLink network.

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("Panlink-network");

int
main(int argc, char* argv[])
{
    bool udp = true;
    bool downlink{true};
    bool useRts{false};
    bool useExtendedBlockAck{false};
    Time simulationTime{"10s"};
    double distance{1.0}; // meters
    double frequency{5};  // whether 2.4, 5 or 6 GHz
    std::size_t nStations{1};
    std::string dlAckSeqType{"NO-OFDMA"};
    bool enableUlOfdma{false};
    bool enableBsrp{false};
    int mcs = 11; // -1 indicates an unset value
    uint32_t payloadSize =
        700; // must fit in the max TX duration when transmitting at MCS 0 over an RU of 26 tones
    std::string phyModel{"Yans"};
    double minExpectedThroughput{0};
    double maxExpectedThroughput{0};
    Time accessReqInterval{0};

    CommandLine cmd(__FILE__);
    cmd.AddValue("frequency",
                 "Whether working in the 2.4, 5 or 6 GHz band (other values gets rejected)",
                 frequency);
    cmd.AddValue("distance",
                 "Distance in meters between the station and the access point",
                 distance);
    cmd.AddValue("simulationTime", "Simulation time", simulationTime);
    cmd.AddValue("udp", "UDP if set to 1, TCP otherwise", udp);
    cmd.AddValue("downlink",
                 "Generate downlink flows if set to 1, uplink flows otherwise",
                 downlink);
    cmd.AddValue("useRts", "Enable/disable RTS/CTS", useRts);
    cmd.AddValue("useExtendedBlockAck", "Enable/disable use of extended BACK", useExtendedBlockAck);
    cmd.AddValue("nStations", "Number of non-AP HE stations", nStations);
    cmd.AddValue("dlAckType",
                 "Ack sequence type for DL OFDMA (NO-OFDMA, ACK-SU-FORMAT, MU-BAR, AGGR-MU-BAR)",
                 dlAckSeqType);
    cmd.AddValue("enableUlOfdma",
                 "Enable UL OFDMA (useful if DL OFDMA is enabled and TCP is used)",
                 enableUlOfdma);
    cmd.AddValue("enableBsrp",
                 "Enable BSRP (useful if DL and UL OFDMA are enabled and TCP is used)",
                 enableBsrp);
    cmd.AddValue(
        "muSchedAccessReqInterval",
        "Duration of the interval between two requests for channel access made by the MU scheduler",
        accessReqInterval);
    cmd.AddValue("mcs", "if set, limit testing to a specific MCS (0-11)", mcs);
    cmd.AddValue("payloadSize", "The application payload size in bytes", payloadSize);
    cmd.AddValue("phyModel",
                 "PHY model to use when OFDMA is disabled (Yans or Spectrum). If OFDMA is enabled "
                 "then Spectrum is automatically selected",
                 phyModel);
    cmd.AddValue("minExpectedThroughput",
                 "if set, simulation fails if the lowest throughput is below this value",
                 minExpectedThroughput);
    cmd.AddValue("maxExpectedThroughput",
                 "if set, simulation fails if the highest throughput is above this value",
                 maxExpectedThroughput);
    cmd.Parse(argc, argv);

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
        Config::SetDefault("ns3::WifiDefaultProtectionManager::EnableMuRts", BooleanValue(true));
    }

    if (dlAckSeqType == "ACK-SU-FORMAT")
    {
        Config::SetDefault("ns3::WifiDefaultAckManager::DlMuAckSequenceType",
                           EnumValue(WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE));
    }
    else if (dlAckSeqType == "MU-BAR")
    {
        Config::SetDefault("ns3::WifiDefaultAckManager::DlMuAckSequenceType",
                           EnumValue(WifiAcknowledgment::DL_MU_TF_MU_BAR));
    }
    else if (dlAckSeqType == "AGGR-MU-BAR")
    {
        Config::SetDefault("ns3::WifiDefaultAckManager::DlMuAckSequenceType",
                           EnumValue(WifiAcknowledgment::DL_MU_AGGREGATE_TF));
    }
    else if (dlAckSeqType != "NO-OFDMA")
    {
        NS_ABORT_MSG("Invalid DL ack sequence type (must be NO-OFDMA, ACK-SU-FORMAT, MU-BAR or "
                     "AGGR-MU-BAR)");
    }

    if (phyModel != "Yans" && phyModel != "Spectrum")
    {
        NS_ABORT_MSG("Invalid PHY model (must be Yans or Spectrum)");
    }
    if (dlAckSeqType != "NO-OFDMA")
    {
        // SpectrumWifiPhy is required for OFDMA
        phyModel = "Spectrum";
    }

    int channelWidth = 80;
    int gi = 800;
    if (!udp)
    {
        Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(payloadSize));
    }

    NodeContainer wifiStaNodes;
    wifiStaNodes.Create(nStations);
    NodeContainer wifiApNode;
    wifiApNode.Create(1);

    NetDeviceContainer apDevice;
    NetDeviceContainer staDevices;
    WifiMacHelper mac;
    WifiHelper wifi;
    std::string channelStr("{0, " + std::to_string(channelWidth) + ", ");
    StringValue ctrlRate;
    auto nonHtRefRateMbps = HePhy::GetNonHtReferenceRate(mcs) / 1e6;

    std::ostringstream ossDataMode;
    ossDataMode << "HeMcs" << mcs;

    if (frequency == 6)
    {
        wifi.SetStandard(WIFI_STANDARD_80211ax);
        ctrlRate = StringValue(ossDataMode.str());
        channelStr += "BAND_6GHZ, 0}";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss",
                            DoubleValue(48));
    }
    else if (frequency == 5)
    {
        wifi.SetStandard(WIFI_STANDARD_80211ax);
        std::ostringstream ossControlMode;
        ossControlMode << "OfdmRate" << nonHtRefRateMbps << "Mbps";
        ctrlRate = StringValue(ossControlMode.str());
        channelStr += "BAND_5GHZ, 0}";
    }
    else if (frequency == 2.4)
    {
        wifi.SetStandard(WIFI_STANDARD_80211ax);
        std::ostringstream ossControlMode;
        ossControlMode << "ErpOfdmRate" << nonHtRefRateMbps << "Mbps";
        ctrlRate = StringValue(ossControlMode.str());
        channelStr += "BAND_2_4GHZ, 0}";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss",
                            DoubleValue(40));
    }
    else
    {
        NS_FATAL_ERROR("Wrong frequency value!");
    }

    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                    "DataMode",
                                    StringValue(ossDataMode.str()),
                                    "ControlMode",
                                    ctrlRate);
    // Set guard interval
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(gi)));

    Ssid ssid = Ssid("ns3-80211ax");

    
    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    YansWifiPhyHelper phy;
    phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    phy.SetChannel(channel.Create());

    mac.SetType("ns3::StaWifiMac",
                "Ssid",
                SsidValue(ssid),
                "MpduBufferSize",
                UintegerValue(useExtendedBlockAck ? 256 : 64));
    phy.Set("ChannelSettings", StringValue(channelStr));
    staDevices = wifi.Install(phy, mac, wifiStaNodes);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter",
                BooleanValue(false),
                "Ssid",
                SsidValue(ssid));
    apDevice = wifi.Install(phy, mac, wifiApNode);

    int64_t streamNumber = 150;
    streamNumber += wifi.AssignStreams(apDevice, streamNumber);
    streamNumber += wifi.AssignStreams(staDevices, streamNumber);

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(distance, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNodes);

    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNodes);
    streamNumber += stack.AssignStreams(wifiApNode, streamNumber);
    streamNumber += stack.AssignStreams(wifiStaNodes, streamNumber);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterfaces;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterfaces = address.Assign(staDevices);
    apNodeInterface = address.Assign(apDevice);

    /* Setting applications */
    ApplicationContainer serverApp;
    auto serverNodes = downlink ? std::ref(wifiStaNodes) : std::ref(wifiApNode);
    Ipv4InterfaceContainer serverInterfaces;
    NodeContainer clientNodes;
    for (std::size_t i = 0; i < nStations; i++)
    {
        serverInterfaces.Add(downlink ? staNodeInterfaces.Get(i)
                                        : apNodeInterface.Get(0));
        clientNodes.Add(downlink ? wifiApNode.Get(0) : wifiStaNodes.Get(i));
    }

    const auto maxLoad = HePhy::GetDataRate(mcs, channelWidth, gi, 1) / nStations;
    // UDP flow
    uint16_t port = 9;
    UdpServerHelper server(port);
    serverApp = server.Install(serverNodes.get());
    streamNumber += server.AssignStreams(serverNodes.get(), streamNumber);

    serverApp.Start(Seconds(0.0));
    serverApp.Stop(simulationTime + Seconds(1.0));
    const auto packetInterval = payloadSize * 8.0 / maxLoad;

    for (std::size_t i = 0; i < nStations; i++)
    {
        UdpClientHelper client(serverInterfaces.GetAddress(i), port);
        client.SetAttribute("MaxPackets", UintegerValue(4294967295U));
        client.SetAttribute("Interval", TimeValue(Seconds(packetInterval)));
        client.SetAttribute("PacketSize", UintegerValue(payloadSize));
        ApplicationContainer clientApp = client.Install(clientNodes.Get(i));
        streamNumber += client.AssignStreams(clientNodes.Get(i), streamNumber);

        clientApp.Start(Seconds(1.0));
        clientApp.Stop(simulationTime + Seconds(1.0));
    }

    Simulator::Schedule(Seconds(0), &Ipv4GlobalRoutingHelper::PopulateRoutingTables);

    Simulator::Stop(simulationTime + Seconds(1.0));
    Simulator::Run();

    std::cout << "ns-3 simulation ends." << std::endl;
    
    return 0;
}
