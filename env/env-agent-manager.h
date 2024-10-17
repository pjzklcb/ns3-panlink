/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_AGENT_MANAGER_H
#define ENV_AGENT_MANAGER_H

#include "env-agent.h"
#include "cca-env-agent.h"

#include "ns3/ptr.h"
#include "ns3/singleton.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvAgentManager : public Singleton<EnvAgentManager>
{
  public:
    /**
     * Register EnvAgents with the specified node.
     * 
     * \param node the bound node
     * \param agentList the list of agents
     */
    void RegisterEnvAgents(Ptr<Node> node, std::vector<AgentType> agentList);

  private:
    /**
     * Create EnvAgent with given type and bound it to the specified node.
     * 
     * \param node the bound node
     * \param type the type of the agent
     * \param memoryId the memory ID assigned to the node
     */
    Ptr<EnvAgent> CreateEnvAgent(Ptr<Node> node, AgentType type, uint32_t memoryId);

    /**
     * Execute information exchange for the specified agent.
     * 
     * \param nodeId the node ID
     * \param type the type of the agent
     * \param obs the obs from the environment
     */
    void DoEnvInfoExchange(uint32_t nodeId, AgentType type, void* obs);
    
    std::map<uint32_t /* Node ID */, uint32_t /* memory ID */> m_memoryMap;
    std::map<uint32_t /* Node ID */, std::map<AgentType, Ptr<EnvAgent>>> m_agentMap;
};

} // namespace ns3

#endif /* ENV_AGENT_MANAGER_H */
