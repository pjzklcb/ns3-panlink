/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "channel-access-env-agent.h"

#include "ns3/panlink-txop.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("ChannelAccessEnvAgent");

NS_OBJECT_ENSURE_REGISTERED(ChannelAccessEnvAgent);

TypeId
ChannelAccessEnvAgent::GetTypeId()
{
    static TypeId tid = TypeId("ns3::ChannelAccessEnvAgent")
                            .SetParent<EnvAgent>()
                            .SetGroupName("Panlink")
                            .AddConstructor<ChannelAccessEnvAgent>();
    return tid;
}

ChannelAccessEnvAgent::ChannelAccessEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

ChannelAccessEnvAgent::~ChannelAccessEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

ChannelAccessEnvAgent::ChannelAccessEnvAgent(Ptr<Node> node, uint32_t memoryId)
    : EnvAgent(node, memoryId)
{
    NS_LOG_FUNCTION(this);

    m_type = RL_CHANNEL_ACCESS;
}

void
ChannelAccessEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.agentType = m_type;

    // Get obs from sensing MAC
    // auto sensingMac = GetSensingMac();
    // auto linkId = obs.linkId;
    // env.bssRssi = sensingMac->GetMinBssRssi(linkId);
    // env.obssRssi = sensingMac->GetMaxObssRssi(linkId);
}

void
ChannelAccessEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    auto& act = msgInterface->GetPy2CppVector()->at(m_memoryId);
    double newBackoff = act.backoff;

    auto& caObs = static_cast<ChannelAccessObs&>(obs);
    auto txop = StaticCast<PanlinkTxop>(GetWifiMac()->GetQosTxop(caObs.ac));
    if (txop->GetBackoffSlots(caObs.linkId) != 0)
    {
        NS_LOG_DEBUG("reset backoff from " << txop->GetBackoffSlots(caObs.linkId) << " to " << newBackoff << " slots");
    }
    else
    {
        NS_LOG_DEBUG("start backoff=" << newBackoff << " slots");
    }
    txop->SetBackoffSlots(newBackoff, caObs.linkId);
}

} // namespace ns3