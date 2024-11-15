/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef CHANNEL_ACCESS_ENV_AGENT_H
#define CHANNEL_ACCESS_ENV_AGENT_H

#include "env-agent.h"

namespace ns3
{

/**
 * \brief env agent for channel access
 * \ingroup panlink
 */

class ChannelAccessEnvAgent : public EnvAgent
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();
    
    ChannelAccessEnvAgent();
    ~ChannelAccessEnvAgent() override;

    struct ChannelAccessObs : public EnvAgent::ModuleObs
    {
        AcIndex ac;
        uint32_t cw;
        uint32_t nDeferredAccess;
        uint32_t waitedSlots;
    };

    void GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs) override;
    void SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs) override;

  private:
};

} // namespace ns3

#endif /* CHANNEL_ACCESS_ENV_AGENT_H */
