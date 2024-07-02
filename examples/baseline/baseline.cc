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
#include "ns3/panlink-app-installer.h"
#include "ns3/panlink-utils.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/udp-server.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-acknowledgment.h"

#include <functional>

// This is an example modified from the examples/wireless/wifi-he-network.cc as the main file for
// the panlink channel access verification.

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("panlink-channel-access");

int
main(int argc, char* argv[])
{
    uint32_t seed = 10; // random seed
    Time simulationTime = Seconds(3);
    bool downlink{true};
    bool useRts{false};
    bool useExtendedBlockAck{false};
    double distance{1.0}; // meters
    double frequency{5};  // whether 2.4, 5 or 6 GHz
    std::size_t nStations{1};
    std::string dlAckSeqType{"NO-OFDMA"};
    bool enableUlOfdma{false};
    bool enableBsrp{false};
    int mcs{11}; // -1 indicates an unset value
    uint32_t payloadSize =
        700; // must fit in the max TX duration when transmitting at MCS 0 over an RU of 26 tones
    Time accessReqInterval{0};
    int gi = 800;

    std::string configPath = "contrib/panlink/examples/configs/";
    std::string nodesConfig = configPath + "nodes.json";
    std::string appConfig = configPath + "application.json";
    bool pcap = false;
    bool verbose = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("seed", "The random seed for this example to run", seed);
    cmd.AddValue("simulationTime", "Total simulation time", simulationTime);
    cmd.AddValue("frequency",
                 "Whether working in the 2.4, 5 or 6 GHz band (other values gets rejected)",
                 frequency);
    cmd.AddValue("distance",
                 "Distance in meters between the station and the access point",
                 distance);
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
    cmd.AddValue("pcap", "Enable generating pcap files", pcap);
    cmd.AddValue("verbose", "Enable printing logs", verbose);
    cmd.Parse(argc, argv);

    WifiHelper g_wifi;
    WifiMacHelper g_mac;
    SpectrumWifiPhyHelper g_phy;
    InternetStackHelper g_inet;
    Ipv4AddressHelper g_ipv4addr;
    MobilityHelper g_mobility;
    NodeContainer g_nodes;
    NetDeviceContainer g_devices;
    Ipv4InterfaceContainer g_interfaces;
    SrcAppMap g_srcAppMap;

    g_mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    //==================================================================//
    //                     Read Configuration Files                     ||
    //==================================================================//
    BssMap bssMap = ReadBssMapFromJson(nodesConfig);
    AppMap appMap = ReadAppInfoFromJson(appConfig);

    //==================================================================//
    //                     Configure Wi-Fi Network                      ||
    //==================================================================//

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

    Config::SetDefault("ns3::WifiMac::MpduBufferSize",
                       UintegerValue(useExtendedBlockAck ? 256 : 64));

    StringValue ctrlRate;
    auto nonHtRefRateMbps = HePhy::GetNonHtReferenceRate(mcs) / 1e6;

    std::ostringstream ossDataMode;
    ossDataMode << "HeMcs" << mcs;

    g_wifi.SetStandard(WIFI_STANDARD_80211ax);
    std::string freqStr;
    if (frequency == 6)
    {
        ctrlRate = StringValue(ossDataMode.str());
        freqStr = "BAND_6GHZ";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue(48));
    }
    else if (frequency == 5)
    {
        std::ostringstream ossControlMode;
        ossControlMode << "OfdmRate" << nonHtRefRateMbps << "Mbps";
        ctrlRate = StringValue(ossControlMode.str());
        freqStr = "BAND_5GHZ";
    }
    else if (frequency == 2.4)
    {
        std::ostringstream ossControlMode;
        ossControlMode << "ErpOfdmRate" << nonHtRefRateMbps << "Mbps";
        ctrlRate = StringValue(ossControlMode.str());
        freqStr = "BAND_2_4GHZ";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue(40));
    }
    else
    {
        NS_FATAL_ERROR("Wrong frequency value!");
    }

    // Set guard interval
    g_wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(gi)));

    if (dlAckSeqType != "NO-OFDMA")
    {
        g_mac.SetMultiUserScheduler("ns3::RrMultiUserScheduler",
                                    "EnableUlOfdma",
                                    BooleanValue(enableUlOfdma),
                                    "EnableBsrp",
                                    BooleanValue(enableBsrp),
                                    "AccessReqInterval",
                                    TimeValue(accessReqInterval));
    }

    // Set channel
    /*
     * SingleModelSpectrumChannel cannot be used with 802.11ax because two
     * spectrum models are required: one with 78.125 kHz bands for HE PPDUs
     * and one with 312.5 kHz bands for, e.g., non-HT PPDUs (for more details,
     * see issue #408 (CLOSED))
     */
    Ptr<MultiModelSpectrumChannel> spectrumChannel = CreateObject<MultiModelSpectrumChannel>();

    Ptr<LogDistancePropagationLossModel> lossModel =
        CreateObject<LogDistancePropagationLossModel>();
    spectrumChannel->AddPropagationLossModel(lossModel);

    g_phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    g_phy.SetChannel(spectrumChannel);

    //==================================================================//
    //                       Deploy Wi-Fi Network                       ||
    //==================================================================//
    for (auto& [bssIdx, bssInfo] : bssMap)
    {
        Ssid ssid = Ssid(bssInfo.m_ssid);
        const char* bssIpBase = bssInfo.m_bssIp.data();

        /* Create nodes */
        NodeContainer bssNodes;
        NetDeviceContainer bssDevices;
        Ipv4InterfaceContainer bssInterfaces;

        bssNodes.Create(bssInfo.m_nodes.size());
        Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

        /* Config each node */
        for (uint32_t i = 0; i < bssInfo.m_nodes.size(); i++)
        {
            PanlinkNode& node = bssInfo.m_nodes[i];
            uint32_t channel = node.m_channel;
            uint32_t bandwidth = node.m_bandwidth;
            std::string channelStr("{" + std::to_string(channel) + ", " +
                                   std::to_string(bandwidth) + ", " + freqStr + ", 0}");

            /* Install Wi-Fi stack */
            g_wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                           "DataMode",
                                           StringValue(ossDataMode.str()),
                                           "ControlMode",
                                           ctrlRate);

            if (node.m_type == "PO")
            {
                g_mac.SetType("ns3::ApWifiMac",
                              "Ssid",
                              SsidValue(ssid),
                              "EnableBeaconJitter",
                              BooleanValue(true));
            }
            else if (node.m_type == "PC")
            {
                g_mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
            }
            else if (node.m_type == "AP")
            {
                g_mac.SetType("ns3::ApWifiMac",
                              "Ssid",
                              SsidValue(ssid),
                              "EnableBeaconJitter",
                              BooleanValue(true));
            }
            else if (node.m_type == "STA")
            {
                g_mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
            }
            else
            {
                NS_ABORT_MSG("Node type not recognized!");
            }

            g_phy.Set("ChannelSettings", StringValue(channelStr));

            bssDevices.Add(g_wifi.Install(g_phy, g_mac, bssNodes.Get(i)));

            /* Config position */
            positionAlloc->Add(Vector(node.m_xPosition, node.m_yPosition, node.m_zPosition));

            /* Record applications */
            for (auto& app : node.m_applications)
            {
                auto it = g_srcAppMap.find(node.m_id);
                if (it == g_srcAppMap.end())
                {
                    g_srcAppMap[node.m_id] = std::vector<SrcApp>{app};
                }
                else
                {
                    g_srcAppMap[node.m_id].push_back(app);
                }
            }
        }

        /* Install Internet stack */
        g_inet.Install(bssNodes);
        g_ipv4addr.SetBase(bssIpBase, "255.255.255.0");
        bssInterfaces = g_ipv4addr.Assign(bssDevices);

        /* Install mobility */
        g_mobility.SetPositionAllocator(positionAlloc);
        g_mobility.Install(bssNodes);

        // Add to the global containers
        g_nodes.Add(bssNodes);
        g_devices.Add(bssDevices);
        g_interfaces.Add(bssInterfaces);
    }

    // Assign random stream number
    int64_t streamNumber = 150;
    streamNumber += g_wifi.AssignStreams(g_devices, streamNumber);
    streamNumber += g_inet.AssignStreams(g_nodes, streamNumber);

    /* Install applications */
    PanlinkAppInstaller appInstaller(appMap);
    appInstaller.Install(g_nodes, g_interfaces, g_srcAppMap);

    Simulator::Schedule(Seconds(0), &Ipv4GlobalRoutingHelper::PopulateRoutingTables);

    Simulator::Stop(simulationTime + Seconds(1.0));
    Simulator::Run();

    // Statistic process

    Simulator::Destroy();

    return 0;
}
