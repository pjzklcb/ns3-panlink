/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "channel-access-env-agent.h"

#include "ns3/panlink-txop.h"
#include "ns3/simulator.h"

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
    m_type = RL_CHANNEL_ACCESS;
}

ChannelAccessEnvAgent::~ChannelAccessEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

void
ChannelAccessEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    const auto now = Simulator::Now();

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.agentType = m_type;

    // Get obs from sensing MAC
    auto sensingMac = GetSensingMac();
    uint32_t n10 = sensingMac->GetNStaTxOkDelayLessThan(MilliSeconds(10), obs.linkId);
    uint32_t n50 = sensingMac->GetNStaTxOkDelayLessThan(MilliSeconds(50), obs.linkId);
    uint32_t n100 = sensingMac->GetNStaTxOkDelayLessThan(MilliSeconds(100), obs.linkId);
    uint32_t n500 = sensingMac->GetNStaTxOkDelayLessThan(MilliSeconds(500), obs.linkId);
    env.nTxOkDelay10ms = n10;
    env.nTxOkDelay50ms = n50 - n10;
    env.nTxOkDelay100ms = n100 - n50;
    env.nTxOkDelay500ms = n500 - n100;
    auto info = sensingMac->GetLastAccessInfo(obs.linkId);
    env.nInQueuePackets = info.nInQueuePackets;
    env.nExpiredPackets = info.nExpiredPackets;
    env.nDroppedPackets = info.nDroppedPackets;
    env.nSendingPackets = info.nSendingPackets;
    env.nAckedPackets = info.nAckedPackets;
    env.nLowDelayPackets = info.nLowDelayPackets;
    env.nHighDelayPackets = info.nHighDelayPackets;

    // Get module obs
    auto& channelAccessObs = static_cast<ChannelAccessObs&>(obs);
    env.ac = static_cast<uint8_t>(channelAccessObs.ac);
    env.cw = channelAccessObs.cw;
    env.nDeferredAccess = channelAccessObs.nDeferredAccess;
    env.waitedSlots = channelAccessObs.waitedSlots;
}

void
ChannelAccessEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    auto& act = msgInterface->GetPy2CppVector()->at(m_memoryId);
    double newBackoff = act.backoff;

    AcIndex ac = static_cast<ChannelAccessObs&>(obs).ac;
    auto txop = StaticCast<PanlinkTxop>(GetWifiMac()->GetQosTxop(ac));
    if (txop->GetBackoffSlots(obs.linkId) != 0)
    {
        NS_LOG_DEBUG("reset backoff from " << txop->GetBackoffSlots(obs.linkId) << " to "
                                           << newBackoff << " slots");
    }
    else
    {
        NS_LOG_DEBUG("start backoff=" << newBackoff << " slots");
    }
    txop->SetBackoffSlots(newBackoff, obs.linkId);
}

} // namespace ns3