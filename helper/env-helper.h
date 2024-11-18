/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_HELPER_H
#define ENV_HELPER_H

#include "ns3/env-agent-manager.h"
#include "ns3/env-monitor.h"
#include "ns3/ipv4-interface-container.h"
#include "ns3/panlink-utils.h"

#include <string>

namespace ns3
{

/**
 * \ingroup panlink
 */

class EnvHelper
{
  public:
    EnvHelper();
    virtual ~EnvHelper();

    EnvHelper(std::string timestamp);

    /**
     * Initialize the environment monitor.
     *
     * \param nodes the nodes
     * \param devices the net devices
     * \param ipv4Interfaces the IPv4 interfaces
     * \param srcApps the source applications
     */
    void InitEnvMonitor(const NodeContainer& nodes,
                        const NetDeviceContainer& devices,
                        const Ipv4InterfaceContainer& ipv4Interfaces,
                        const SrcAppMap& srcApps);

    /**
     * Initialize the environment agent manager.
     *
     * \param map the agent map
     */
    void InitEnvAgentManager(AgentMap map);

    /**
     * Schedule an initial information exchange.
     *
     * \param startTime the start time
     */
    void ScheduleInitInfoExchange(Time startTime);

    /**
     * Save the simulation results to the given directory.
     * 
     * \param resultDir the directory
     */
    void SaveResults(std::string resultDir);

  private:
    std::string m_timestamp;
};

} // namespace ns3

#endif /* ENV_HELPER_H */