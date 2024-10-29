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

#include "app-installer.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("AppInstaller");

void
AppInstaller::ReadAppDictFromJson(std::string filename)
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
    rapidjson::Document document;
    document.Parse(json_str.c_str());

    if (document.HasParseError())
    {
        NS_LOG_ERROR("Failed to parse " << filename << ": "
                                        << rapidjson::GetParseErrorFunc(document.GetParseError()));
    }

    if (!document.IsObject() || !document.HasMember("applications") ||
        !document["applications"].IsObject())
    {
        NS_LOG_ERROR("Failed to parse: app_dict.json has invalid applications.");
    }

    for (const auto& app : document["applications"].GetArray())
    {
        std::string appName = app["name"].GetString();
        if (auto it = m_appDict.find(appName); it != m_appDict.end())
        {
            NS_LOG_ERROR("Duplicated application name: " << appName << " in " << filename);
        }

        AppInfo appInfo;
        appInfo.m_payloadInterval = ParseRandomVariableStream(app["payload_interval"]);
        appInfo.m_payloadSize = ParseRandomVariableStream(app["payload_size"]);
        appInfo.m_fragmentSize = app["fragment_size"].GetInt();

        m_appDict[appName] = appInfo;
    }
}

std::string
AppInstaller::ParseRandomVariableStream(const rapidjson::Value& item)
{
    std::string rand;
    if (item.IsDouble())
    {
        rand = "ns3::ConstantRandomVariable[Constant=" + std::to_string(item.GetDouble()) + "]";
        return rand;
    }
    if (item.IsInt())
    {
        rand = "ns3::ConstantRandomVariable[Constant=" + std::to_string(item.GetInt()) + "]";
        return rand;
    }

    if (!item.IsObject())
    {
        NS_ABORT_MSG("Missing application member in app_dict.json");
        return rand;
    }

    std::string dict = item["dict"].GetString();
    if (dict == "largest_extreme")
    {
        auto location = std::to_string(item["location"].GetDouble());
        auto scale = std::to_string(item["scale"].GetDouble());
        rand =
            "ns3::LargestExtremeValueRandomVariable[Location=" + location + "|Scale=" + scale + "]";
    }
    else
    {
        NS_LOG_ERROR("Unknown distribution in app_dict.json");
    }

    return rand;
}

ApplicationContainer
AppInstaller::InstallApplications(const NodeContainer& nodes,
                                  const Ipv4InterfaceContainer& interfaces,
                                  const SrcAppMap& srcApps)
{
    NS_LOG_FUNCTION(this);

    ApplicationContainer apps;

    for (auto [srcNodeId, srcAppList] : srcApps)
    {
        for (size_t i = 0; i < srcAppList.size(); i++)
        {
            auto srcApp = srcAppList[i];
            std::string appName = srcApp.m_name;
            if (auto it = m_appDict.find(appName); it == m_appDict.end())
            {
                NS_ABORT_MSG("Unknown application-" << appName);
            }

            uint32_t dstNodeId = srcApp.m_dst;
            uint16_t port = srcApp.m_port;

            InetSocketAddress dstAddr = InetSocketAddress(interfaces.GetAddress(dstNodeId), port);
            ApplicationContainer app = DoInstallApplication(nodes.Get(srcNodeId), dstAddr, srcApp);
            apps.Add(app);

            std::cout << "Node-" << srcNodeId << " installed application \"" << appName << "\" to node-"
                      << dstNodeId << "(IP: " << interfaces.GetAddress(dstNodeId)
                      << ", Port:" << port << ")" << std::endl;

            // Record the sink app
            m_sinkAppList.insert({dstNodeId, port});
        }
    }

    apps.Add(InstallSinkApplications(nodes));

    return apps;
}

ApplicationContainer
AppInstaller::DoInstallApplication(Ptr<Node> node, InetSocketAddress dstAddr, SrcApp srcApp)
{
    NS_LOG_FUNCTION(this);

    AppInfo appInfo = m_appDict[srcApp.m_name];
    uint16_t tos = AcIndexToTos(srcApp.m_ac);

    BurstHelper burstHelper("ns3::UdpSocketFactory", dstAddr);
    burstHelper.SetAttribute("FragmentSize", UintegerValue(appInfo.m_fragmentSize));
    burstHelper.SetAttribute("Tos", UintegerValue(tos));
    burstHelper.SetBurstGenerator("ns3::SimpleBurstGenerator",
                                   "PeriodRv",
                                   StringValue(appInfo.m_payloadInterval),
                                   "BurstSizeRv",
                                   StringValue(appInfo.m_payloadSize));

    m_appHelpers.push_back(burstHelper);
    auto app = burstHelper.Install(node);
    app.Start(Seconds(srcApp.m_startTime));
    app.Stop(Seconds(srcApp.m_endTime));

    return app;
}

ApplicationContainer
AppInstaller::InstallSinkApplications(NodeContainer nodes)
{
    NS_LOG_FUNCTION(this);

    ApplicationContainer apps;
    for (auto [dstNodeId, port] : m_sinkAppList)
    {
        BurstSinkHelper sink("ns3::UdpSocketFactory",
                             InetSocketAddress(Ipv4Address::GetAny(), port));
        auto sinkApp = sink.Install(nodes.Get(dstNodeId));
        sinkApp.Start(Seconds(0.0));
        sinkApp.Stop(Simulator::GetMaximumSimulationTime());

        m_appHelpers.push_back(sink);
        apps.Add(sinkApp);
    }

    return apps;
}

int64_t
AppInstaller::AssignStreams(NodeContainer nodes, int64_t stream)
{
    NS_LOG_FUNCTION(this);

    for (auto helper : m_appHelpers)
    {
        stream += helper.AssignStreams(nodes, stream);
    }

    return stream;
}

} // namespace ns3
