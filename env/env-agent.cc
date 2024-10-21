/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "env-agent.h"

#include "ns3/enum.h"
#include "ns3/simulator.h"

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

void 
EnvAgent::SetPeriod(Time period)
{
    NS_LOG_FUNCTION(this << period);
    m_period = period;
}

AgentType
EnvAgent::GetAgentType() const
{
    return m_type;
}

Ptr<Node>
EnvAgent::GetNode() const
{
    return m_node;
}

Time
EnvAgent::GetPeriod() const
{
    return m_period;
}

void 
EnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, void* obs)
{
    NS_LOG_FUNCTION(this);

    m_counter += 1;
    Time now = Simulator::Now();

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.nodeId = m_node->GetId();
    env.agentType = m_type;
    env.counter = m_counter;
    env.simTime = now.GetSeconds();
}

void
EnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface)
{
    NS_LOG_FUNCTION(this);
    
    // auto& act = msgInterface->GetCpp2PyVector()->at(m_memoryId);
}

} // namespace ns3