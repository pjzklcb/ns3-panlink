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
#include "ns3/uinteger.h"

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
            .AddAttribute("MemoryId",
                          "The memory ID of this env agent.",
                          TypeId::ATTR_GET |
                              TypeId::ATTR_CONSTRUCT, // prevent setting after construction
                          UintegerValue(0),
                          MakeUintegerAccessor(&EnvAgent::GetMemoryId, &EnvAgent::SetMemoryId),
                          MakeUintegerChecker<uint32_t>())
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
EnvAgent::DoDispose()
{
    NS_LOG_FUNCTION(this);

    m_node = nullptr;
    Object::DoDispose();
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
EnvAgent::SetMemoryId(uint32_t memoryId)
{
    NS_LOG_FUNCTION(this << memoryId);
    m_memoryId = memoryId;
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

uint32_t
EnvAgent::GetMemoryId() const
{
    return m_memoryId;
}

Time
EnvAgent::GetPeriod() const
{
    return m_period;
}

Ptr<WifiMac>
EnvAgent::GetWifiMac() const
{
    NS_LOG_FUNCTION(this);

    for (size_t i = 0; i < m_node->GetNDevices(); ++i)
    {
        if (auto device = DynamicCast<WifiNetDevice>(m_node->GetDevice(i)))
        {
            return device->GetMac();
        }
    }
    return 0;
}

Ptr<SensingMac>
EnvAgent::GetSensingMac() const
{
    NS_LOG_FUNCTION(this);

    auto mac = GetWifiMac();
    // if (auto apMac = DynamicCast<PanlinkApWifiMac>(mac))
    // {
    //     return apMac->GetSensingMac();
    // }
    if (auto staMac = DynamicCast<PanlinkStaWifiMac>(mac))
    {
        return staMac->GetSensingMac();
    }
    return 0;
}

void
EnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    NS_ABORT_MSG_IF(m_node == nullptr, "Env agent must bind to a node");

    m_counter += 1;
    Time now = Simulator::Now();

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.nodeId = m_node->GetId();
    env.agentType = m_type;
    env.counter = m_counter;
    env.simTime = now.GetSeconds();

    // Get obs from sensing MAC if it exists
    if (auto sensingMac = GetSensingMac())
    {
        auto linkId = obs.linkId;
        int i = 0;
        for (const auto& ac : {AC_BE, AC_BK, AC_VI, AC_VO})
        {
            env.thptReq[i] = sensingMac->GetThptReq(m_period, ac);
            env.thpt[i] = sensingMac->GetThpt(m_period, ac);
            env.delayReq[i] = 30; // ms
            env.delay[i] = sensingMac->GetDelay(m_period, ac);
            i++;
        }
        env.nBssStas = sensingMac->GetNBssStas(linkId);
        env.nObssStas = sensingMac->GetNObssStas(linkId);
        env.selfChannelUtil = sensingMac->GetSelfChannelUtil(m_period, linkId);
        env.otherChannelUtil = sensingMac->GetOtherChannelUtil(m_period, linkId);
    }
}

void
EnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface)
{
    NS_LOG_FUNCTION(this);

    // auto& act = msgInterface->GetPy2CppVector()->at(m_memoryId);
}

} // namespace ns3