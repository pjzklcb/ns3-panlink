/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent.h"

#include "ns3/enum.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("EnvAgent");

NS_OBJECT_ENSURE_REGISTERED(EnvAgent);

EnvAgent::~EnvAgent()
{
}

EnvAgent::EnvAgent(Ptr<Node> node, uint32_t memoryId)
    : m_node(node),
      m_memoryId(memoryId),
      m_counter(0)
{
    NS_LOG_FUNCTION(this);
}

void
EnvAgent::SetAgentType(AgentType type)
{
    NS_LOG_FUNCTION(this << type);
    m_type = type;
}

AgentType
EnvAgent::GetAgentType() const
{
    return m_type;
}

void 
EnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, void* info)
{
    NS_LOG_FUNCTION(this);
}

void
EnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface)
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3