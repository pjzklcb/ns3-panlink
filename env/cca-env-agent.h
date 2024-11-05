/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef CCA_ENV_AGENT_H
#define CCA_ENV_AGENT_H

#include "env-agent.h"

namespace ns3
{

/**
 * \brief env agent for cca sensitivity
 * \ingroup panlink
 */

class CcaEnvAgent : public EnvAgent
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    
    CcaEnvAgent();
    ~CcaEnvAgent() override;

    CcaEnvAgent(Ptr<Node> node, uint32_t memoryId);

    struct CcaModuleObs : public ModuleObs
    {
        double apRssi;
        double bssRssi;
        double obssRssi;
    };

    ModuleObs GetModuleObs(Time period) const override;
    void GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs) override;
    void SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface) override;

  private:
};

} // namespace ns3

#endif /* ENV_AGENT_H */
