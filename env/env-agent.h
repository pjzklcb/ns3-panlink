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

#include "ns3/env-utils.h"
#include "ns3/node.h"
#include "ns3/ns3-ai-msg-interface.h"
#include "ns3/object.h"

namespace ns3
{

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
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    EnvAgent();
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
     * Set the bound node.
     *
     * \param node the bound node
     */
    void SetNode(Ptr<Node> node);
    /**
     * Set the memory ID of the bound node.
     *
     * \param memoryId the memory ID
     */
    void SetMemoryId(uint32_t memoryId);
    /**
     * Set the info exchange period of the node (0 for trigger-based).
     *
     * \param period the info exchange period
     */
    void SetPeriod(Time period);

    /**
     * Get the type of this env agent
     */
    AgentType GetAgentType() const;
    /**
     * Get the bound node.
     */
    Ptr<Node> GetNode() const;
    /**
     * Get the memory ID of the bound node.
     */
    uint32_t GetMemoryId() const;
    /**
     * Get the info exchange period of the node.
     */
    Time GetPeriod() const;

    /**
     * Get the observation of environment.
     *
     * \param msgInterface the message interface
     * \param obs the observation from the environment
     */
    virtual void GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, void* obs);
    /**
     * Set the action feedbacked from RL agent.
     */
    virtual void SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface);

  protected:
    AgentType m_type;    //!< the type of this env agent
    Ptr<Node> m_node;    //!< the bound node
    uint32_t m_memoryId; //!< memory ID assigned to the bound node
    uint32_t m_counter;  //!< interaction counter
    Time m_period;       //!< the info exchange period (0 for trigger-based)
};

} // namespace ns3

#endif /* ENV_AGENT_H */
