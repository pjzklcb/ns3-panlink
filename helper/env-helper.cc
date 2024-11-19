/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-helper.h"

#include "ns3/log.h"
#include "ns3/node-list.h"
#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/simulator.h"

#include <filesystem>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvHelper");

EnvHelper::EnvHelper()
    : EnvHelper("My")
{
    NS_LOG_FUNCTION(this);
}

EnvHelper::~EnvHelper()
{
}

EnvHelper::EnvHelper(std::string timestamp)
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
EnvHelper::InitEnvMonitor(const NodeContainer& nodes,
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
EnvHelper::Install(const EnvAgentHelper& agentHelper, Ptr<Node> node) const
{
    NS_LOG_FUNCTION(this << node);

    EnvAgentManager::Get()->Register(agentHelper.Create(node));
}

void
EnvHelper::ScheduleInitInfoExchange(Time startTime)
{
    NS_LOG_FUNCTION(this);

    auto agentManager = EnvAgentManager::Get();
    // Register dummy agent if there is no agent
    if (agentManager->IsEmpty())
    {
        agentManager->InstallDummyAgent();
    }

    // Schedule an initial information exchange
    Simulator::Schedule(startTime, &EnvAgentManager::ScheduleInitInfoExchange, agentManager);
}

void
EnvHelper::SaveResults(std::string resultDir)
{
    std::filesystem::create_directory(resultDir + "csv/");

    EnvMonitor::Get()->WriteResultsToCsv(resultDir + "csv/");
}

} // namespace ns3