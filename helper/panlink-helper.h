/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_HELPER_H
#define PANLINK_HELPER_H

#include "ns3/panlink-utils.h"
#include <string>

namespace ns3
{

/**
 * \ingroup panlink
 */

class PanlinkHelper
{
  public:
    PanlinkHelper();
    virtual ~PanlinkHelper();

    PanlinkHelper(std::string timestamp);

    /**
     * Initialize the environment agent manager
     * 
     * \param panlinkInfo the list of Panlink nodes
     */
    void InitEnvAgentManager(std::vector<PanlinkNodeInfo> panlinkInfo);

    /**
     * Schedule an initial information exchange.
     * 
     * \param startTime the start time
     */
    void ScheduleInitInfoExchange(Time startTime);

  private:
    std::string m_timestamp;
};

} // namespace ns3

#endif /* PANLINK_HELPER_H */