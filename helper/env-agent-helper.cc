/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent-helper.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgentHelper");

EnvAgentHelper::EnvAgentHelper()
{
    NS_LOG_FUNCTION(this);
}

EnvAgentHelper::~EnvAgentHelper()
{
    NS_LOG_FUNCTION(this);
}

Ptr<EnvAgent> 
EnvAgentHelper::Create(Ptr<Node> node) const
{
    NS_LOG_FUNCTION(this << node);

    ObjectFactory agentFactory = m_agent;
    auto agent = agentFactory.Create<EnvAgent>();
    agent->SetNode(node);

    return agent;
}

std::string 
EnvAgentHelper::GetAgentName(AgentType type) const
{
    NS_LOG_FUNCTION(this << type);

    switch (type)
    {
    case AgentType::NO_RL:
        return "ns3::EnvAgent";
    case AgentType::RL_CCA_SENSITIVITY:
        return "ns3::CcaEnvAgent";
    default:
        NS_ABORT_MSG("Unrecognized agent type");
        break;
    }
}

} // namespace ns3