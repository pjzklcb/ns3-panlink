/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent-manager.h"

#include "ns3/env-agent-helper.h"
#include "ns3/node-list.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgentManager");

EnvAgentManager::~EnvAgentManager()
{
    NS_LOG_FUNCTION(this);

    for (auto& [nodeId, agentList] : m_agentMap)
    {
        for (auto& [type, agent] : agentList)
        {
            agent = nullptr;
        }
    }
    m_agentMap.clear();
}

uint32_t
EnvAgentManager::GetMemoryId(Ptr<Node> node)
{
    NS_LOG_FUNCTION(this << node);

    return GetMemoryId(node->GetId());
}

uint32_t
EnvAgentManager::GetMemoryId(uint32_t nodeId)
{
    NS_LOG_FUNCTION(this << nodeId);

    uint32_t memoryId;

    auto it = m_memoryMap.find(nodeId);
    if (it == m_memoryMap.end())
    {
        // assign the memory ID of new node
        memoryId = m_memoryMap.size();
        m_memoryMap[nodeId] = memoryId;
        m_agentMap[nodeId] = std::map<AgentType, Ptr<EnvAgent>>();
    }
    else
    {
        memoryId = it->second;
    }

    return memoryId;
}

void
EnvAgentManager::Register(Ptr<EnvAgent> agent)
{
    NS_LOG_FUNCTION(this << agent);

    auto node = agent->GetNode();
    auto type = agent->GetAgentType();
    m_agentMap[node->GetId()][type] = agent;
}

Ptr<EnvAgent>
EnvAgentManager::GetEnvAgent(uint32_t nodeId, AgentType type)
{
    NS_LOG_FUNCTION(this << nodeId << type);

    auto it = m_agentMap[nodeId].find(type);
    if (it == m_agentMap[nodeId].end())
    {
        NS_LOG_DEBUG("Node-" << nodeId << " does not have an agent of type " << type);
        return 0;
    }
    return it->second;
}

bool
EnvAgentManager::IsEmpty()
{
    NS_LOG_FUNCTION(this);
    return m_memoryMap.size() == 0;
}

void
EnvAgentManager::InstallDummyAgent()
{
    NS_LOG_FUNCTION(this);

    EnvAgentHelper helper;
    helper.SetType("ns3::EnvAgent", "MemoryId", UintegerValue(0), "Period", TimeValue(Seconds(1)));
    auto dummyAgent = helper.Create(NodeList::GetNode(0));

    m_agentMap[0] = std::map<AgentType, Ptr<EnvAgent>>();
    m_agentMap[0][AgentType::NO_RL] = dummyAgent;

    std::cout << "Dummy agent installed." << std::endl;
}

void
EnvAgentManager::ScheduleInitInfoExchange()
{
    NS_LOG_FUNCTION(this);
    for (auto [nodeId, agentList] : m_agentMap)
    {
        for (auto [type, agent] : agentList)
        {
            auto period = agent->GetPeriod();
            if (period == Seconds(0))
            {
                // Skip trigger-based agent
                continue;
            }

            // Schedule an information exchange
            Simulator::Schedule(period,
                                &EnvAgentManager::ScheduleInfoExchange,
                                this,
                                nodeId,
                                type,
                                period);
        }
    }
}

void
EnvAgentManager::ScheduleInfoExchange(uint32_t nodeId, AgentType type, Time period)
{
    NS_LOG_FUNCTION(this << nodeId << type << period);

    auto obs = EnvAgent::ModuleObs();
    EnvInfoExchange(nodeId, type, obs);

    // Periodically trigger information exchange
    Simulator::Schedule(period, &EnvAgentManager::ScheduleInfoExchange, this, nodeId, type, period);
}

void
EnvAgentManager::EnvInfoExchange(uint32_t nodeId, AgentType type, EnvAgent::ModuleObs& obs)
{
    NS_LOG_FUNCTION(this << nodeId << type);

    Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface =
        Ns3AiMsgInterface::Get()->GetInterface<Obs, Act>();

    auto agent = GetEnvAgent(nodeId, type);
    NS_ABORT_MSG_IF(agent == nullptr, "Env agent not found");

    do
    {
        msgInterface->CppSendBegin();
        agent->GetEnvObs(msgInterface, obs);
        msgInterface->CppSendEnd();
    } while (agent->HasMoreData());

    msgInterface->CppRecvBegin();
    agent->SetEnvAct(msgInterface, obs);
    msgInterface->CppRecvEnd();
}

} // namespace ns3