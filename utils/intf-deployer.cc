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
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "intf-deployer.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("IntfDeployer");

IntfDeployer::IntfDeployer()
{
    NS_LOG_FUNCTION(this);
}

IntfDeployer::~IntfDeployer()
{
    NS_LOG_FUNCTION(this);
}

void
IntfDeployer::ReadIntfConfigFile(std::string filename)
{
    NS_LOG_FUNCTION(this << filename);

    // Open the JSON file
    std::ifstream file(filename);
    if (!file.is_open())
    {
        NS_LOG_ERROR("Failed to open " << filename);
    }

    // Read the JSON file into a string
    std::string json_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Parse the JSON string
    m_document.Parse(json_str.c_str());

    if (m_document.HasParseError() || !m_document.IsObject())
    {
        NS_LOG_ERROR("Failed to parse "
                     << filename << ": "
                     << rapidjson::GetParseErrorFunc(m_document.GetParseError()));
    }

    if (!m_document.HasMember("ap_list_len") || !m_document.HasMember("probe_list_len") ||
        !m_document.HasMember("ap_list") || !m_document.HasMember("probe_list"))
    {
        NS_LOG_ERROR("Missing key in interference config file.");
    }
}

void
IntfDeployer::DeployInterference(WifiHelper& wifiHelper,
                                 WifiMacHelper& macHelper,
                                 SpectrumWifiPhyHelper& phyHelper,
                                 InternetStackHelper& inetHelper,
                                 Ipv4AddressHelper& ipv4Helper,
                                 MobilityHelper& mobilityHelper,
                                 SrcAppMap& srcAppMap)
{
    NS_LOG_FUNCTION(this);

    auto nApList = m_document["ap_list_len"].GetInt();
    auto nProbeList = m_document["probe_list_len"].GetInt();
    rapidjson::Value& apList = m_document["ap_list"];
    rapidjson::Value& probeList = m_document["probe_list"];
    NS_ABORT_MSG_IF(nApList != static_cast<int>(apList.Size()) ||
                        nProbeList != static_cast<int>(probeList.Size()),
                    "Invalid interference config file");

    // ====================================
    // Deploy AP and associated STAs
    // ====================================
    for (int i = 0; i < nApList; i++)
    {
        /* Parse config */
        auto nStaList = apList[i]["sta_list_len"].GetInt();
        std::map<uint32_t, rapidjson::Value> nodeList;
        std::map<uint32_t, int> staIndexMap; // the mapping from sta_idx to idx in nodeList
        nodeList[0] = apList[i];
        for (int j = 0; j < nStaList; j++)
        {
            rapidjson::Value& sta = apList[i]["sta_list"][j];
            nodeList[sta["sta_idx"].GetInt()] = sta;
            staIndexMap[sta["sta_idx"].GetInt()] = j + 1;
        }

        NodeContainer nodes;
        NetDeviceContainer devices;
        Ipv4InterfaceContainer interfaces;
        Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

        /* Create Nodes */
        nodes.Create(1 + nStaList);
        Ssid ssid = Ssid("intf-" + nodes.Get(0)->GetId());
        for (int j = 0; j < 1 + nStaList; j++)
        {
            auto& node = nodeList[j];
            uint32_t channel = node["channel"].GetInt();
            uint32_t bandwidth = node["bandwidth"].GetInt();
            std::string channelStr("{" + std::to_string(channel) + ", " +
                                   std::to_string(bandwidth) + ", BAND_5GHZ, 0}");

            /* Install Wi-Fi stack */
            wifiHelper.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                               "DataMode",
                                               StringValue("HeMcs9"),
                                               "ControlMode",
                                               StringValue("HeMcs0"));

            if (j == 0)
            {
                macHelper.SetType("ns3::ApWifiMac",
                                  "Ssid",
                                  SsidValue(ssid),
                                  "EnableBeaconJitter",
                                  BooleanValue(true));
            }
            else
            {
                macHelper.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
            }

            phyHelper.Set("ChannelSettings", StringValue(channelStr));

            devices.Add(wifiHelper.Install(phyHelper, macHelper, nodes.Get(j)));

            /* Config position */
            positionAlloc->Add(Vector(node["position"]["x"].GetDouble(),
                                      node["position"]["y"].GetDouble(),
                                      node["position"]["z"].GetDouble()));

            /* Record applications */
            if (node["application_list_len"].GetInt() > 0)
            {
                srcAppMap[nodes.Get(j)->GetId()] = std::vector<SrcApp>();
                for (int k = 0; k < node["application_list_len"].GetInt(); k++)
                {
                    auto& app = node["application_list"][k];
                    uint32_t dst = nodes.Get(0)->GetId(); // AP node ID
                    if (j == 0)
                    {
                        dst = nodes.Get(staIndexMap[app["dst"].GetInt()])->GetId();
                    }

                    srcAppMap[nodes.Get(j)->GetId()].emplace_back(
                        SrcApp{app["name"].GetString(),
                               dst,
                               50000 + uint(k),
                               StringToAcIndex(app["qos"].GetString()),
                               app["start_time"].GetDouble(),
                               app["end_time"].GetDouble()});
                }
            }
        }

        /* Install mobility */
        mobilityHelper.SetPositionAllocator(positionAlloc);
        mobilityHelper.Install(nodes);

        /* Internet stack*/
        inetHelper.Install(nodes);
        std::string base = "10.144." + std::to_string(i) + ".0";
        ipv4Helper.SetBase(base.data(), "255.255.255.0");
        interfaces = ipv4Helper.Assign(devices);

        m_nodes.Add(nodes);
        m_devices.Add(devices);
        m_interfaces.Add(interfaces);
    }

    // ====================================
    // Deploy Probing STAs
    // ====================================
    NodeContainer nodes;
    NetDeviceContainer devices;
    Ipv4InterfaceContainer interfaces;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    /* Create Nodes */
    nodes.Create(nProbeList);
    for (int i = 0; i < nProbeList; i++)
    {
        auto& probe = probeList[i];
        uint32_t channel = probe["channel"].GetInt();
        uint32_t bandwidth = probe["bandwidth"].GetInt();
        std::string channelStr("{" + std::to_string(channel) + ", " + std::to_string(bandwidth) +
                               ", BAND_5GHZ, 0}");

        /* Install Wi-Fi stack */
        wifiHelper.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                           "DataMode",
                                           StringValue("HeMcs9"),
                                           "ControlMode",
                                           StringValue("HeMcs0"));
        macHelper.SetType("ns3::StaWifiMac",
                          "Ssid",
                          SsidValue(), // Wildcard SSID
                          "ActiveProbing",
                          BooleanValue(true),
                          "ProbeDelay",
                          StringValue("ns3::ConstantRandomVariable[Constant=" +
                                      std::to_string(probe["probe_delay"].GetDouble()) +
                                      "]")); // in microseconds
        phyHelper.Set("ChannelSettings", StringValue(channelStr));

        devices.Add(wifiHelper.Install(phyHelper, macHelper, nodes.Get(i)));

        /* Config position */
        positionAlloc->Add(Vector(probe["position"]["x"].GetDouble(),
                                  probe["position"]["y"].GetDouble(),
                                  probe["position"]["z"].GetDouble()));
    }

    /* Install mobility */
    mobilityHelper.SetPositionAllocator(positionAlloc);
    mobilityHelper.Install(nodes);

    /* Internet stack*/
    inetHelper.Install(nodes);
    for (int i = 0; i < nProbeList; i++)
    {
        std::string base = "10.1." + std::to_string(i / 255) + "." + std::to_string(i % 255);
        ipv4Helper.SetBase(base.data(), "255.255.255.0");
        interfaces.Add(ipv4Helper.Assign(devices.Get(i)));
    }

    m_nodes.Add(nodes);
    m_devices.Add(devices);
    m_interfaces.Add(interfaces);
}

NodeContainer
IntfDeployer::GetNodes() const
{
    return m_nodes;
}

NetDeviceContainer
IntfDeployer::GetDevices() const
{
    return m_devices;
}

Ipv4InterfaceContainer
IntfDeployer::GetInterfaces() const
{
    return m_interfaces;
}

} // namespace ns3
