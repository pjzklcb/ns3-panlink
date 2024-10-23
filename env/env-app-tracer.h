/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_APP_TRACER_H
#define ENV_APP_TRACER_H

#include "env-utils.h"

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/seq-ts-size-frag-header.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvAppTracer : public Object
{
  public:
    /**
     * Hanlde the packet creation traceback.
     *
     * \param map the in-flight packet map
     * \param nodeId the node ID
     * \param dstPort the destination port
     * \param packet the created packet
     */
    void HandleAppTx(InFlightPacketMap& map,
                     uint32_t nodeId,
                     uint16_t dstPort,
                     Ptr<const Packet> packet);
    /**
     * Handle the packet reception traceback.
     *
     * \param map the in-flight packet map
     * \param packet the received packet
     */
    void HandleAppRx(InFlightPacketMap& map, Ptr<const Packet> packet);

    /**
     * Handle the burst creation traceback.
     *
     * \param map the in-flight packet map
     * \param burstMap the map decirbes all fragments in burst
     * \param nodeId the node ID
     * \param dstPort the destination port
     * \param burst the created burst
     * \param header the header of the created burst
     */
    void HandleBurstTx(BurstMap& burstMap,
                       uint32_t nodeId,
                       uint16_t dstPort,
                       Ptr<const Packet> burst,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the burst reception traceback.
     *
     * \param map the in-flight packet map
     * \param burstMap the map decirbes all fragments in burst
     * \param nodeId the node ID
     * \param dstPort the destination port
     * \param burst the received burst
     * \param header the header of the created burst
     */
    void HandleBurstRx(InFlightPacketMap& map,
                       BurstMap& burstMap,
                       uint32_t nodeId,
                       uint16_t dstPort,
                       Ptr<const Packet> burst,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation creation traceback.
     *
     * \param map the in-flight packet map
     * \param burstMap the map decirbes all fragments in burst
     * \param nodeId the node ID
     * \param dstPort the destination port
     * \param fragment the created fragment
     * \param header the header of the created fragment
     */
    void HandleFragmentTx(InFlightPacketMap& map,
                          BurstMap& burstMap,
                          uint32_t nodeId,
                          uint16_t dstPort,
                          Ptr<const Packet> fragment,
                          const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation reception traceback.
     *
     * \param map the in-flight packet map
     * \param fragment the received fragment
     * \param header the header of the received fragment
     */
    void HandleFragmentRx(InFlightPacketMap& map,
                          Ptr<const Packet> fragment,
                          const SeqTsSizeFragHeader& header);

  private:
    std::map<uint32_t /* Node ID */, std::vector<std::pair<Time, double>>> m_e2eDelay;
};

} // namespace ns3

#endif /* ENV_APP_TRACER_H */