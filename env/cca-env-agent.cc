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

EnvAgent::ModuleObs
CcaEnvAgent::GetModuleObs(Time period) const
{
    NS_LOG_FUNCTION(this << period);

    CcaModuleObs obs;
    if (auto device = DynamicCast<WifiNetDevice>(m_node->GetDevice(0)))
    {
        if (auto staMac = DynamicCast<PanlinkStaWifiMac>(device->GetMac()))
        {
            auto sensingMac = staMac->GetSensingMac();
            int i = 0;
            for (const auto& [ac, _] : wifiAcList)
            {
                obs.thptReq[i] = sensingMac->GetThptReq(period, ac);
                obs.thpt[i] = sensingMac->GetThpt(period, ac);
                obs.delay[i] = sensingMac->GetDelay(period, ac);
                i++;
            }
            obs.apRssi = 0;
            obs.bssRssi = 0;
            obs.obssRssi = 0;
        }
    }

    return obs;
}

void
CcaEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    auto ccaObs = static_cast<CcaModuleObs>(obs);

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.apRssi = ccaObs.apRssi;
    env.bssRssi = ccaObs.bssRssi;
    env.obssRssi = ccaObs.obssRssi;
}

void
CcaEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface)
{
    NS_LOG_FUNCTION(this);

    // auto& act = msgInterface->GetCpp2PyVector()->at(m_memoryId);
}

} // namespace ns3