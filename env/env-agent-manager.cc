/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent-manager.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgentManager");

void 
EnvAgentManager::RegisterEnvAgents(Ptr<Node> node, std::vector<AgentType> agentList)
{
    NS_LOG_FUNCTION(this << node << agentList);

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
    for (auto type : agentList)
    {
        uint32_t memoryId = m_memoryMap[node->GetId()];
        m_agentMap[node->GetId()][type] = CreateEnvAgent(node, type, memoryId);
        std::cout << "Node-" << node->GetId() << " has an agent of type " << type << std::endl;
    }
}

Ptr<EnvAgent>
EnvAgentManager::CreateEnvAgent(Ptr<Node> node, AgentType type, uint32_t memoryId)
{
    NS_LOG_FUNCTION(this << node << type << memoryId);
    
    Ptr<EnvAgent> agent;
    switch (type)
    {
    case AgentType::RL_CCA_SENSITIVITY:
        agent = CreateObject<CcaEnvAgent>(node, memoryId);
        break;
    
    default:
        NS_LOG_DEBUG("Unrecognized agent type");
        break;
    }
    
    agent->SetAgentType(type);
    
    return agent;
}

void 
EnvAgentManager::DoEnvInfoExchange(uint32_t nodeId, AgentType type, void* obs)
{
    NS_LOG_FUNCTION(this << nodeId << type);

    auto it = m_agentMap[nodeId].find(type);
    if (it == m_agentMap[nodeId].end())
    {
        NS_LOG_DEBUG("Node-" << nodeId << " does not have an agent of type " << type);
        return;
    }

    auto agent = it->second;
    agent->GetEnvObs(obs);
    agent->SetEnvAct();
}

}