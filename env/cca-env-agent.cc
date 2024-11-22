/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "cca-env-agent.h"

#include "ns3/double.h"
#include "ns3/threshold-preamble-detection-model.h"
#include "ns3/wifi-phy.h"

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
    m_type = RL_CCA_SENSITIVITY;
}

CcaEnvAgent::~CcaEnvAgent()
{
    NS_LOG_FUNCTION(this);
}

void
CcaEnvAgent::GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    EnvAgent::GetEnvObs(msgInterface, obs);

    auto& env = msgInterface->GetCpp2PyVector()->at(m_memoryId);
    env.agentType = m_type;

    // Get obs from sensing MAC
    auto sensingMac = GetSensingMac();
    auto linkId = obs.linkId;
    int i = 0;
    for (const auto& aci : {AC_BE, AC_BK, AC_VI, AC_VO})
    {
        env.thptReq[i] = sensingMac->GetThptReq(m_period, aci);
        env.thpt[i] = sensingMac->GetThpt(m_period, aci);
        env.delayReq[i] = 30; // ms
        env.delay[i] = sensingMac->GetDelay(m_period, aci);
        i++;
    }
    env.bssRssi = sensingMac->GetMinBssRssi(linkId);
    env.obssRssi = sensingMac->GetMaxObssRssi(linkId);
}

void
CcaEnvAgent::SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs)
{
    NS_LOG_FUNCTION(this);

    auto& act = msgInterface->GetPy2CppVector()->at(m_memoryId);
    double newCca = act.ccaSensitivity;

    auto preambleDetectionModel = CreateObject<ThresholdPreambleDetectionModel>();
    preambleDetectionModel->SetAttribute("MinimumRssi", DoubleValue(newCca));

    auto phy = GetWifiMac()->GetWifiPhy(obs.linkId);
    phy->SetCcaSensitivityThreshold(newCca);
    phy->SetPreambleDetectionModel(preambleDetectionModel);
}

} // namespace ns3