/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent-manager.h"

#include "ns3/node-list.h"
#include "ns3/simulator.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgentManager");

void 
EnvAgentManager::RegisterEnvAgents(Ptr<Node> node, std::vector<AgentInfo> agentList)
{
    NS_LOG_FUNCTION(this << node);

    if (agentList.size() == 0)
    {
        std::cout << "Node-" << node->GetId() << ": Empty agent list." << std::endl;
        return;
    }

    // assign the memory ID of new node
    if (auto it = m_memoryMap.find(node->GetId()); it == m_memoryMap.end())
    {
        uint32_t memoryId = m_memoryMap.size();
        m_memoryMap[node->GetId()] = memoryId;
        m_agentMap[node->GetId()] = std::map<AgentType, Ptr<EnvAgent>>();
    }

    // create EnvAgent
    for (auto agentInfo : agentList)
    {
        auto type = agentInfo.m_type;
        uint32_t memoryId = m_memoryMap[node->GetId()];
        m_agentMap[node->GetId()][type] = CreateEnvAgent(node, agentInfo, memoryId);
        std::cout << "Node-" << node->GetId() << " installs an agent with type=" << type << std::endl;
    }
}

bool 
EnvAgentManager::IsEmpty()
{
    NS_LOG_FUNCTION(this);
    return m_memoryMap.size() == 0;
}

void 
EnvAgentManager::RegisterDummyAgent()
{
    NS_LOG_FUNCTION(this);

    m_agentMap[0] = std::map<AgentType, Ptr<EnvAgent>>();
    auto dummyAgent = CreateEnvAgent(NodeList::GetNode(0), AgentInfo(AgentType::NO_RL, Seconds(1)), 0);
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
            Simulator::Schedule(period, &EnvAgentManager::ScheduleInfoExchange, this, nodeId, type, period);
        }
    }
}

void 
EnvAgentManager::ScheduleInfoExchange(uint32_t nodeId, AgentType type, Time period)
{
    NS_LOG_FUNCTION(this << nodeId << type << period);
    
    DoEnvInfoExchange(nodeId, type, nullptr);

    // Periodically trigger information exchange
    Simulator::Schedule(period, &EnvAgentManager::ScheduleInfoExchange, this, nodeId, type, period);
}

Ptr<EnvAgent>
EnvAgentManager::CreateEnvAgent(Ptr<Node> node, AgentInfo agentInfo, uint32_t memoryId)
{
    NS_LOG_FUNCTION(this << node << memoryId);
    
    Ptr<EnvAgent> agent;
    switch (agentInfo.m_type)
    {
    case AgentType::NO_RL:
        agent = CreateObject<EnvAgent>(node, memoryId);
        break;
        
    case AgentType::RL_CCA_SENSITIVITY:
        agent = CreateObject<CcaEnvAgent>(node, memoryId);
        break;
    
    default:
        NS_LOG_DEBUG("Unrecognized agent type");
        break;
    }
    
    agent->SetAgentType(agentInfo.m_type);
    agent->SetPeriod(agentInfo.m_period);
    
    return agent;
}

void 
EnvAgentManager::DoEnvInfoExchange(uint32_t nodeId, AgentType type, void* obs)
{
    NS_LOG_FUNCTION(this << nodeId << type);

    Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface = 
        Ns3AiMsgInterface::Get()->GetInterface<Obs, Act>();

    auto it = m_agentMap[nodeId].find(type);
    if (it == m_agentMap[nodeId].end())
    {
        NS_LOG_DEBUG("Node-" << nodeId << " does not have an agent of type " << type);
        return;
    }

    auto agent = it->second;

    msgInterface->CppSendBegin();
    agent->GetEnvObs(msgInterface, obs);
    msgInterface->CppSendEnd();

    msgInterface->CppRecvBegin();
    agent->SetEnvAct(msgInterface);
    msgInterface->CppRecvEnd();
}

}