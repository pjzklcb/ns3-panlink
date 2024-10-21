/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_PHY_TRACER_H
#define ENV_PHY_TRACER_H

#include "env-utils.h"

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/packet.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvPhyTracer : public Object
{
  public:
    /**
     * Handle the packet transmission begin traceback.
     *
     * \param map the in-flight packet map
     * \param nodeId the source node ID
     * \param p the packet
     * \param txPowerW the tx power
     */
    void HandlePhyTxBegin(InFlightPacketMap& map,
                          uint32_t nodeId,
                          Ptr<const Packet> p,
                          double txPowerW);
    /**
     * Handle the packet transmission end traceback.
     *
     * \param map the in-flight packet map
     * \param p the packet
     */
    void HandlePhyTxEnd(InFlightPacketMap& map, Ptr<const Packet> p);

  private:
    std::map<uint32_t /* Node ID */, std::vector<std::pair<Time, Time>>> m_airTime;
};

} // namespace ns3

#endif /* ENV_PHY_TRACER_H */