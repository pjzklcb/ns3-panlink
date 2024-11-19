/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_AGENT_HELPER_H
#define ENV_AGENT_HELPER_H

#include "ns3/env-agent.h"
#include "ns3/env-utils.h"
#include "ns3/object-factory.h"

#include <string>

namespace ns3
{

/**
 * @ingroup panlink
 */

class EnvAgentHelper
{
  public:
    EnvAgentHelper();
    virtual ~EnvAgentHelper();

    /**
     * @tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
     * @param type the type of ns3::EnvAgent to create.
     * @param args A sequence of name-value pairs of the attributes to set.
     *
     * All the attributes specified in this method should exist in the requested EnvAgent.
     */
    template <typename... Args>
    void SetType(std::string type, Args&&... args);

    /**
     * @param node the node on which the Env Agent object will reside
     * @returns a new Env Agent object.
     *
     * This allows the ns3::EnvHelper class to create EnvAgent objects from ns3::EnvHelper::Install.
     */
    virtual Ptr<EnvAgent> Create(Ptr<Node> node) const;

    /**
     * @returns the name of the Env Agent
     */
    std::string GetAgentName(AgentType type) const;

  private:
    ObjectFactory m_agent;
};

} // namespace ns3

/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3
{

template <typename... Args>
void
EnvAgentHelper::SetType(std::string type, Args&&... args)
{
    m_agent.SetTypeId(type);
    m_agent.Set(args...);
}

} // namespace ns3

#endif /* ENV_HELPER_H */