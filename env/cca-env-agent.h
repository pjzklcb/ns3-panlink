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
    // Delete default constructor
    CcaEnvAgent() = delete;
    ~CcaEnvAgent() override;

    CcaEnvAgent(Ptr<Node> node, uint32_t memoryId);

    void GetEnvObs(void* obs) override;
    void SetEnvAct() override;

  private:
};

} // namespace ns3

#endif /* ENV_AGENT_H */
