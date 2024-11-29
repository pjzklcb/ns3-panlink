/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent-helper.h"
#include "ns3/env-agent-manager.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgentHelper");

EnvAgentHelper::EnvAgentHelper()
{
    NS_LOG_FUNCTION(this);

    // create env agent manager instance
    EnvAgentManager::Get();
}

EnvAgentHelper::~EnvAgentHelper()
{
    NS_LOG_FUNCTION(this);
}

Ptr<EnvAgent> 
EnvAgentHelper::Create(Ptr<Node> node) const
{
    NS_LOG_FUNCTION(this << node);

    auto memoryId = EnvAgentManager::Get()->GetMemoryId(node);

    ObjectFactory agentFactory = m_agent;
    agentFactory.Set("MemoryId", UintegerValue(memoryId));
    auto agent = agentFactory.Create<EnvAgent>();
    agent->SetNode(node);

    return agent;
}

} // namespace ns3