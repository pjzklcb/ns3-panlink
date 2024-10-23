/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "ns3/app-installer.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/intf-deployer.h"
#include "ns3/mobility-module.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/network-module.h"
#include "ns3/panlink-helper.h"
#include "ns3/panlink-utils.h"
#include "ns3/wifi-module.h"

#include <filesystem>

// This is an example modified from the examples/wireless/wifi-he-network.cc as the main file for
// the env-agent framework verification.

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("panlink-baseline");

int
main(int argc, char* argv[])
{
    std::string timestamp = "0";
    uint32_t seed = 10; // random seed
    Time simulationTime = Seconds(3);
    bool enableIntf{false};
    bool downlink{true};
    bool useRts{false};
    bool useExtendedBlockAck{false};
    double distance{1.0}; // meters
    double frequency{5};  // whether 2.4, 5 or 6 GHz
    std::size_t nStations{1};
    std::string dlAckSeqType{"NO-OFDMA"};
    bool enableUlOfdma{false};
    bool enableBsrp{false};
    uint32_t payloadSize =
        700; // must fit in the max TX duration when transmitting at MCS 0 over an RU of 26 tones
    Time accessReqInterval{0};
    int gi = 800;

    std::string configPath = "contrib/panlink/configs/";
    std::string panlinkConfig = configPath + "config_panlink.json";
    std::string intfConfig = configPath + "config_intf.json";
    std::string appConfig = configPath + "app_dict.json";
    bool pcap = false;
    bool stats = false;
    bool verbose = false;

    CommandLine cmd(__FILE__);
    cmd.AddValue("timestamp", "The timestamp for this example to run", timestamp);
    cmd.AddValue("seed", "The random seed for this example to run", seed);
    cmd.AddValue("simulationTime", "Total simulation time", simulationTime);
    cmd.AddValue("enableIntf", "Enable interference", enableIntf);
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
    cmd.AddValue("payloadSize", "The application payload size in bytes", payloadSize);
    cmd.AddValue("pcap", "Enable generating pcap files", pcap);
    cmd.AddValue("stats", "Enable results saving", stats);
    cmd.AddValue("verbose", "Enable printing logs", verbose);
    cmd.Parse(argc, argv);

    if (verbose)
    {
        LogComponentEnableAll(LOG_PREFIX_ALL);
        LogComponentEnable("EnvAgent", LOG_LEVEL_ALL);
        LogComponentEnable("EnvAgentManager", LOG_LEVEL_ALL);
    }

    WifiHelper g_wifiHelper;
    WifiMacHelper g_macHelper;
    SpectrumWifiPhyHelper g_phyHelper;
    InternetStackHelper g_inetHelper;
    Ipv4AddressHelper g_ipv4Helper;
    MobilityHelper g_mobilityHelper;
    NodeContainer g_nodes;
    NetDeviceContainer g_devices;
    Ipv4InterfaceContainer g_interfaces;
    SrcAppMap g_srcAppMap;

    g_mobilityHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    //==================================================================//
    //                     Read Configuration Files                     ||
    //==================================================================//
    std::cout << "Reading configuration files..." << std::endl;
    std::vector<PanlinkNodeInfo> g_panlinkInfo = ReadNodesFromJson(panlinkConfig);
    AppInstaller::Get()->ReadAppDictFromJson(appConfig);

    //==================================================================//
    //                      Configure Wi-Fi Stack                       ||
    //==================================================================//

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
        Config::SetDefault("ns3::WifiDefaultProtectionManager::EnableMuRts", BooleanValue(true));
    }

    if (dlAckSeqType == "ACK-SU-FORMAT")
    {
        Config::SetDefault(
            "ns3::WifiDefaultAckManager::DlMuAckSequenceType",
            EnumValue<WifiAcknowledgment::Method>(WifiAcknowledgment::DL_MU_BAR_BA_SEQUENCE));
    }
    else if (dlAckSeqType == "MU-BAR")
    {
        Config::SetDefault(
            "ns3::WifiDefaultAckManager::DlMuAckSequenceType",
            EnumValue<WifiAcknowledgment::Method>(WifiAcknowledgment::DL_MU_TF_MU_BAR));
    }
    else if (dlAckSeqType == "AGGR-MU-BAR")
    {
        Config::SetDefault(
            "ns3::WifiDefaultAckManager::DlMuAckSequenceType",
            EnumValue<WifiAcknowledgment::Method>(WifiAcknowledgment::DL_MU_AGGREGATE_TF));
    }
    else if (dlAckSeqType != "NO-OFDMA")
    {
        NS_ABORT_MSG("Invalid DL ack sequence type (must be NO-OFDMA, ACK-SU-FORMAT, MU-BAR or "
                     "AGGR-MU-BAR)");
    }

    Config::SetDefault("ns3::WifiMac::MpduBufferSize",
                       UintegerValue(useExtendedBlockAck ? 256 : 64));

    g_wifiHelper.SetStandard(WIFI_STANDARD_80211ax);
    std::string freqStr;
    if (frequency == 6)
    {
        freqStr = "BAND_6GHZ";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue(48));
    }
    else if (frequency == 5)
    {
        freqStr = "BAND_5GHZ";
    }
    else if (frequency == 2.4)
    {
        freqStr = "BAND_2_4GHZ";
        Config::SetDefault("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue(40));
    }
    else
    {
        NS_FATAL_ERROR("Wrong frequency value!");
    }

    // Set guard interval
    g_wifiHelper.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(gi)));

    if (dlAckSeqType != "NO-OFDMA")
    {
        g_macHelper.SetMultiUserScheduler("ns3::RrMultiUserScheduler",
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

    g_phyHelper.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    g_phyHelper.SetChannel(spectrumChannel);

    //==================================================================//
    //                       Deploy Panlink Nodes                       ||
    //==================================================================//
    std::cout << "Deploying " << g_panlinkInfo.size() << " nodes..." << std::endl;

    NodeContainer panlinkNodes;
    NetDeviceContainer panlinkDevices;
    Ipv4InterfaceContainer panlinkInterfaces;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    panlinkNodes.Create(g_panlinkInfo.size());
    Ssid ssid = Ssid("panlink");
    for (uint32_t i = 0; i < g_panlinkInfo.size(); i++)
    {
        auto& node = g_panlinkInfo[i];
        uint32_t channel = node.m_channel;
        uint32_t bandwidth = node.m_bandwidth;
        std::string channelStr("{" + std::to_string(channel) + ", " + std::to_string(bandwidth) +
                               ", " + freqStr + ", 0}");

        /* Install Wi-Fi stack */
        g_wifiHelper.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                             "DataMode",
                                             StringValue("HeMcs9"),
                                             "ControlMode",
                                             StringValue("HeMcs0"));

        if (node.m_type == "PO")
        {
            g_macHelper.SetType("ns3::ApWifiMac",
                                "Ssid",
                                SsidValue(ssid),
                                "EnableBeaconJitter",
                                BooleanValue(true));
        }
        else if (node.m_type == "PC")
        {
            g_macHelper.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
        }
        else if (node.m_type == "AP")
        {
            g_macHelper.SetType("ns3::ApWifiMac",
                                "Ssid",
                                SsidValue(ssid),
                                "EnableBeaconJitter",
                                BooleanValue(true));
        }
        else if (node.m_type == "STA")
        {
            g_macHelper.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
        }
        else
        {
            NS_ABORT_MSG("Node type not recognized!");
        }

        g_phyHelper.Set("ChannelSettings", StringValue(channelStr));

        panlinkDevices.Add(g_wifiHelper.Install(g_phyHelper, g_macHelper, panlinkNodes.Get(i)));

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

    /* Install mobility */
    g_mobilityHelper.SetPositionAllocator(positionAlloc);
    g_mobilityHelper.Install(panlinkNodes);

    /* Internet stack*/
    g_inetHelper.Install(panlinkNodes);
    g_ipv4Helper.SetBase("192.168.0.0", "255.255.255.0");
    panlinkInterfaces = g_ipv4Helper.Assign(panlinkDevices);

    g_nodes.Add(panlinkNodes);
    g_devices.Add(panlinkDevices);
    g_interfaces.Add(panlinkInterfaces);

    std::cout << "Panlink nodes deployed." << std::endl;
    //==================================================================//
    //                       Deploy Interference                        ||
    //==================================================================//
    if (enableIntf)
    {
        std::cout << "Deploying interference..." << std::endl;
        IntfDeployer intfDeployer;
        std::cout << "Interference deployed." << std::endl;
    }

    //==================================================================//
    //                         Other Processes                          ||
    //==================================================================//
    Simulator::Schedule(Seconds(0), &Ipv4GlobalRoutingHelper::PopulateRoutingTables);

    int64_t streamNumber = 150;
    streamNumber += g_wifiHelper.AssignStreams(g_devices, streamNumber);
    streamNumber += g_inetHelper.AssignStreams(g_nodes, streamNumber);

    /* Install applications */
    AppInstaller::Get()->InstallApplications(g_nodes, g_interfaces, g_srcAppMap);

    /* Create result dir */
    std::string resultDir = "contrib/panlink/result-" + timestamp + "/";
    if (pcap || stats)
    {
        namespace fs = std::filesystem;
        if (!fs::is_directory(resultDir) || !fs::exists(resultDir))
        {
            fs::create_directory(resultDir);
        }
    }

    /* Create pcap file */
    if (pcap)
    {
        g_phyHelper.EnablePcap(resultDir + g_panlinkInfo[0].m_type, panlinkDevices.Get(0));
    }

    //==================================================================//
    //                         Install Agents                           ||
    //==================================================================//
    PanlinkHelper panlinkHelper(timestamp);
    panlinkHelper.InitEnvMonitor(g_nodes, g_devices, g_interfaces, g_srcAppMap);
    panlinkHelper.InitEnvAgentManager(g_panlinkInfo);
    panlinkHelper.ScheduleInitInfoExchange(Seconds(0.1));

    //==================================================================//
    //                         Run Simulation                           ||
    //==================================================================//
    std::cout << "Start simulation..." << std::endl;
    Simulator::Stop(simulationTime + Seconds(1.0));
    Simulator::Run();

    //==================================================================//
    //                        Statistic Process                         ||
    //==================================================================//

    Simulator::Destroy();
    std::cout << "ns-3 simulation done." << std::endl;

    return 0;
}
