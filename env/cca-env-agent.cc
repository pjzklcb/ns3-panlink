/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "cca-env-agent.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("CcaEnvAgent");

NS_OBJECT_ENSURE_REGISTERED(CcaEnvAgent);

TypeId
CcaEnvAgent::GetTypeId()
{
    static TypeId tid = TypeId("ns3::CcaEnvAgent")
                            .SetParent<EnvAgent>()
                            .SetGroupName("Panlink")
                            .AddConstructor<CcaEnvAgent>();
    return tid;
}

CcaEnvAgent::CcaEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

CcaEnvAgent::~CcaEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

CcaEnvAgent::CcaEnvAgent(Ptr<Node> node, uint32_t memoryId)
    : EnvAgent(node, memoryId)
{
    NS_LOG_FUNCTION(this);

    m_type = RL_CCA_SENSITIVITY;
}

void
CcaEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, void* obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    // auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    // env.nodeId = m_node->GetId();
}

void
CcaEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface)
{
    NS_LOG_FUNCTION(this);

    // auto& act = msgInterface->GetCpp2PyVector()->at(m_memoryId);
}

} // namespace ns3