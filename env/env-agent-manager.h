/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_AGENT_MANAGER_H
#define ENV_AGENT_MANAGER_H

#include "env-agent.h"

#include "ns3/ptr.h"
#include "ns3/singleton.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvAgentManager : public Singleton<EnvAgentManager>
{
  public:
  private:
    std::map<uint32_t /* Node ID */, std::vector<Ptr<EnvAgent>>> m_agents;
};

} // namespace ns3

#endif /* ENV_AGENT_MANAGER_H */
