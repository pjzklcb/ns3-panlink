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

TypeId
EnvAgent::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::EnvAgent")
            .SetParent<Object>()
            .SetGroupName("Panlink")
            .AddConstructor<EnvAgent>()
            .AddAttribute("AgentType",
                          "The type of this env agent.",
                          EnumValue<AgentType>(NO_RL),
                          MakeEnumAccessor<AgentType>(&EnvAgent::m_type),
                          MakeEnumChecker<AgentType>(NO_RL,
                                                     "NO_RL",
                                                     RL_CCA_SENSITIVITY,
                                                     "CCA_SENSITIVITY",
                                                     RL_CHANNEL_ACCESS,
                                                     "CHANNEL_ACCESS",
                                                     RL_RATE_CONTROL,
                                                     "RATE_CONTROL",
                                                     RL_AGGREGATION,
                                                     "AGGREGATION"))
            .AddAttribute("Period",
                          "The interaction period of this env agent (0 for trigger-based).",
                          TimeValue(Seconds(0)),
                          MakeTimeAccessor(&EnvAgent::m_period),
                          MakeTimeChecker());
    return tid;
}

EnvAgent::EnvAgent()
    : m_node(nullptr),
      m_memoryId(0),
      m_counter(0)
{
    NS_LOG_FUNCTION(this);
}

EnvAgent::~EnvAgent()
{
    NS_LOG_FUNCTION(this);
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
EnvAgent::SetNode(Ptr<Node> node)
{
    NS_LOG_FUNCTION(this << node);
    m_node = node;
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

    NS_ABORT_MSG_IF(m_node, "Env agent must bind to a node");

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