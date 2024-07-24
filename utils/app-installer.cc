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

AppInstaller::AppInstaller(AppMap appMap)
    : m_appMap(appMap)
{
    NS_LOG_FUNCTION(this);
}

AppInstaller::~AppInstaller()
{
    NS_LOG_FUNCTION(this);
}

ApplicationContainer
AppInstaller::Install(NodeContainer nodes,
                      Ipv4InterfaceContainer interfaces,
                      SrcAppMap srcApps)
{
    NS_LOG_FUNCTION(this);

    ApplicationContainer apps;

    for (auto [srcNodeId, srcAppList] : srcApps)
    {
        for (auto srcApp : srcAppList)
        {
            std::string appName = srcApp.m_name;
            uint32_t dstNodeId = srcApp.m_dst;
            uint32_t port = srcApp.m_port;

            InetSocketAddress dstAddr = InetSocketAddress(interfaces.GetAddress(dstNodeId), port);
            ApplicationContainer app = Install(nodes.Get(srcNodeId), dstAddr, appName);
            
            app.Start(Seconds(srcApp.m_startTime));
            app.Stop(Seconds(srcApp.m_endTime));
            apps.Add(app);

            std::cout << "Installed application: " << appName 
                      << " from node-: " << srcNodeId << " to node-: " << dstNodeId 
                      << "(IP: " << interfaces.GetAddress(dstNodeId) 
                      << ", Port:" << port << ")" << std::endl;

            // Record the sink app
            auto it = m_appMap.find(appName);
            if (it == m_appMap.end())
            {
                NS_LOG_ERROR("Unknown application: " << appName);
                return apps;
            }

            std::string appType = m_appMap[appName].m_type;
            SinkApp sinkApp = std::make_tuple(dstNodeId, appType, port);
        }
    }

    return apps;
}

ApplicationContainer
AppInstaller::Install(Ptr<Node> node, InetSocketAddress dstAddr, std::string appName)
{
    NS_LOG_FUNCTION(this);

    ApplicationContainer app;

    auto it = m_appMap.find(appName);
    if (it == m_appMap.end())
    {
        NS_LOG_ERROR("Unknown application: " << appName);
        return app;
    }
    
    AppInfo appInfo = m_appMap[appName];
    if (appInfo.m_type == "udp")
    {
        UdpClientHelper client(dstAddr);
        client.SetAttribute("MaxPackets", UintegerValue(appInfo.m_params["MaxPackets"]));
        client.SetAttribute("Interval", TimeValue(Seconds(appInfo.m_params["Interval"])));
        client.SetAttribute("PacketSize", UintegerValue(appInfo.m_params["PacketSize"]));
        app = client.Install(node);

        m_appHelpers.push_back(client);
    }
    else if (appInfo.m_type == "constant")
    {
        std::string interval = std::to_string(appInfo.m_params["Interval"]); // in seconds
        std::string burstSize = std::to_string(appInfo.m_params["BurstSize"]);
        BurstyHelper burstyHelper("ns3::UdpSocketFactory", dstAddr);
        burstyHelper.SetAttribute("FragmentSize", UintegerValue(appInfo.m_params["FragmentSize"]));
        burstyHelper.SetBurstGenerator("ns3::SimpleBurstGenerator",
                                       "PeriodRv",
                                       StringValue("ns3::ConstantRandomVariable[Constant=" + interval + "]"),
                                       "BurstSizeRv",
                                       StringValue("ns3::ConstantRandomVariable[Constant=" + burstSize + "]"));

        // m_appHelpers.push_back(client);
    }
    else if (appInfo.m_type == "poisson")
    {
        // TODO: implement poisson distribution
    }
    else
    {
        NS_LOG_ERROR("Unknown application type: " << appInfo.m_type);
        return app;
    }

    return app;
}

ApplicationContainer 
AppInstaller::InstallSinkApp(NodeContainer nodes)
{
    NS_LOG_FUNCTION(this);

    ApplicationContainer apps;
    for (auto [dstNodeId, appType, port] : m_sinkAppList)
    {
        ApplicationContainer sinkApp;
        if (appType == "udp")
        {
            UdpServerHelper server(port);
            sinkApp = server.Install(nodes.Get(dstNodeId));
            sinkApp.Start(Seconds(0.0));
            sinkApp.Stop(Simulator::GetMaximumSimulationTime());

            m_appHelpers.push_back(server);
        }
        else if (appType == "constant")
        {
            BurstSinkHelper sink("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), port));
            sinkApp = sink.Install(nodes.Get(dstNodeId));
            sinkApp.Start(Seconds(0.0));
            sinkApp.Stop(Simulator::GetMaximumSimulationTime());

            // m_appHelpers.push_back(sink);
        }
        else if (appType == "poisson")
        {
            // TODO: implement poisson
        }
        else
        {
            NS_LOG_ERROR("Unknown application type: " << appType);
            return apps;
        }
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
