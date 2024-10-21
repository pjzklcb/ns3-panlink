/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_PHY_TRACER_H
#define ENV_PHY_TRACER_H

#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/nstime.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvPhyTracer : public Object
{
  public:
    /**
     * Handle the packer transmission begin traceback.
     * 
     * \param context string contains the node ID
     * \param p the packet
     * \param txPowerW the tx power
     */
    void HandlePhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW);
    /**
     * Handle the packer transmission end traceback.
     * 
     * \param context string contains the node ID
     * \param p the packet
     */
    void HandlePhyTxEnd(std::string context, Ptr<const Packet> p);

  private:
    std::map<uint32_t /* Node ID */, std::vector<std::pair<Time, Time>>> m_airTime;
};

} // namespace ns3

#endif /* ENV_PHY_TRACER_H */