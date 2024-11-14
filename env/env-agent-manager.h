/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_AGENT_MANAGER_H
#define ENV_AGENT_MANAGER_H

#include "cca-env-agent.h"
#include "env-agent.h"

#include "ns3/panlink-utils.h"
#include "ns3/ptr.h"
#include "ns3/singleton.h"

namespace ns3
{

/**
 * \ingroup panlink
 * \brief Class to manage environment agents of each node.
 *
 * This class is responsible for creating and managing environment agents for
 * each node. Environment agents are created based on the configurations
 * provided by the user and are bound to the node. The user can ask the
 * manager to schedule an information exchange for all registered agents or
 * for the specified agent.
 */
class EnvAgentManager : public Singleton<EnvAgentManager>
{
  public:
    ~EnvAgentManager();

    /**
     * Install EnvAgents with the specified node.
     *
     * \param node the bound node
     * \param agentList the list of agents
     */
    void Install(Ptr<Node> node, std::vector<AgentInfo> agentList);

    /**
     * Get the environment agent of the specified node and type.
     *
     * \param nodeId the node ID
     * \param type the type of the agent
     */
    Ptr<EnvAgent> GetEnvAgent(uint32_t nodeId, AgentType type);

    /**
     * Check if there is no agent installed.
     */
    bool IsEmpty();

    /**
     * Install a dummy agent.
     */
    void InstallDummyAgent();

    /**
     * Schedule an information exchange for all registered agents that do info
     * exchange periodically.
     */
    void ScheduleInitInfoExchange();

    /**
     * Execute information exchange for the specified agent.
     *
     * \param nodeId the node ID
     * \param type the type of the agent
     * \param obs the obs from the functional module
     */
    void EnvInfoExchange(uint32_t nodeId, AgentType type, EnvAgent::ModuleObs& obs);

  private:
    /**
     * Schedule an information exchange for the specified agent.
     *
     * \param nodeId the node ID
     * \param type the type of the agent
     * \param period the info exchange period of the agent
     */
    void ScheduleInfoExchange(uint32_t nodeId, AgentType type, Time period);

    std::map<uint32_t /* Node ID */, uint32_t /* memory ID */> m_memoryMap;
    std::map<uint32_t /* Node ID */, std::map<AgentType, Ptr<EnvAgent>>> m_agentMap;
};


} // namespace ns3

#endif /* ENV_AGENT_MANAGER_H */
