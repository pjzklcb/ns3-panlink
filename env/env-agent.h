/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_AGENT_H
#define ENV_AGENT_H

#include "obs-act-list.h"

#include "ns3/node.h"
#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/object.h"

namespace ns3
{

enum AgentType
{
    NO_RL = 0,
    RL_CCA_SENSITIVITY,
    RL_CHANNEL_ACCESS,
    RL_RATE_CONTROL,
    RL_AGGREGATION
};

/**
 * \brief base class for all env agents.
 * \ingroup panlink
 *
 * This class serves as the base class for environment agents that collect
 * observations and implement actions in the ns-3 simulation environment.
 */

class EnvAgent : public Object
{
  public:
    // Delete default constructor
    EnvAgent() = delete;
    ~EnvAgent() override;
    
    /**
     * EnvAgent must bind to a node and specify the corresponding memory ID.
     * 
     * \param node the bound node
     * \param memoryId the memory ID 
     */
    EnvAgent(Ptr<Node> node, uint32_t memoryId);

    // Delete copy constructor and assignment operator to avoid misuse
    EnvAgent(const EnvAgent&) = delete;
    EnvAgent& operator=(const EnvAgent&) = delete;

    /**
     * Set the type of this env agent.
     *
     * \param type the type of this env agent
     */
    void SetAgentType(AgentType type);
    /**
     * Get the type of this env agent
     */
    AgentType GetAgentType() const;

    /**
     * Get the observation of environment.
     */
    void GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, void* info);
    /**
     * Set the action feedbacked from RL agent.
     */
    void SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface);

  protected:
    AgentType m_type;    //!< the type of this env agent
    Ptr<Node> m_node;    //!< the bound node
    uint32_t m_memoryId; //!< memory ID assigned to the bound node
    uint32_t m_counter;  //!< interaction counter
};

} // namespace ns3

#endif /* ENV_AGENT_H */
