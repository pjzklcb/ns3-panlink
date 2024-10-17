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

CcaEnvAgent::~CcaEnvAgent()
{
}

CcaEnvAgent::CcaEnvAgent(Ptr<Node> node, uint32_t memoryId)
    : EnvAgent(node, memoryId)
{
    NS_LOG_FUNCTION(this);
}

void 
CcaEnvAgent::GetEnvObs(void* obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(obs);

    // Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface = 
    //     Ns3AiMsgInterface::Get()->GetInterface<Obs, Act>();

    // msgInterface->CppSendBegin();
    // auto env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    // env.nodeId = m_node->GetId();
    // msgInterface->CppSendEnd();
}

void
CcaEnvAgent::SetEnvAct()
{
    NS_LOG_FUNCTION(this);

    // Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface = 
    //     Ns3AiMsgInterface::Get()->GetInterface<Obs, Act>();
    
    // msgInterface->CppRecvBegin();
    // auto act = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    // msgInterface->CppRecvEnd();
}

} // namespace ns3