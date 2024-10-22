/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "panlink-helper.h"

#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkHelper");

PanlinkHelper::PanlinkHelper()
    : PanlinkHelper("My")
{
    NS_LOG_FUNCTION(this);
}

PanlinkHelper::~PanlinkHelper()
{
}

PanlinkHelper::PanlinkHelper(std::string timestamp)
    : m_timestamp(timestamp)
{
    NS_LOG_FUNCTION(this);

    // Create Ns3AiMsgInterface instance
    auto msgInterface = Ns3AiMsgInterface::Get();
    msgInterface->SetIsMemoryCreator(false);
    msgInterface->SetUseVector(true);
    msgInterface->SetHandleFinish(true);

    msgInterface->SetNames(m_timestamp + " Seg",
                           m_timestamp + " Cpp to Python Msg",
                           m_timestamp + " Python to Cpp Msg",
                           m_timestamp + " Lockable");
}

void
PanlinkHelper::InitEnvMonitor(const NodeContainer& nodes,
                              const NetDeviceContainer& devices,
                              const Ipv4InterfaceContainer& ipv4Interfaces,
                              const SrcAppMap& srcApps)
{
    NS_LOG_FUNCTION(this);

    // Create EnvMonitor instance
    auto envMonitor = EnvMonitor::Get();
    envMonitor->SetNodes(nodes);
    envMonitor->SetDevices(devices);
    envMonitor->SetIpv4Interfaces(ipv4Interfaces);
    envMonitor->SetSrcApps(srcApps);

    // Connect all trace sources
    envMonitor->ConnectTraceSources();
}

void
PanlinkHelper::InitEnvAgentManager(std::vector<PanlinkNodeInfo> panlinkInfo)
{
    NS_LOG_FUNCTION(this);

    // Create EnvAgentManager instance
    auto agentManager = EnvAgentManager::Get();

    for (auto node : panlinkInfo)
    {
        agentManager->RegisterEnvAgents(NodeList::GetNode(node.m_id), node.m_agents);
    }
}

void
PanlinkHelper::ScheduleInitInfoExchange(Time startTime)
{
    NS_LOG_FUNCTION(this);

    auto agentManager = EnvAgentManager::Get();
    // Register dummy agent if there is no agent
    if (agentManager->IsEmpty())
    {
        agentManager->RegisterDummyAgent();
    }

    // Schedule an initial information exchange
    Simulator::Schedule(startTime, &EnvAgentManager::ScheduleInitInfoExchange, agentManager);
}

} // namespace ns3