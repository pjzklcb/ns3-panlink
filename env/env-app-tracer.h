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
     * \param packet the created packet
     * \param dstPort the destination port
     */
    void HandleAppTx(InFlightPacketMap& map,
                     uint32_t nodeId,
                     Ptr<const Packet> packet,
                     uint16_t dstPort);
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
     * \param burstMap the burst map indicates contained fragments
     * \param burst the created burst
     * \param header the header of the created burst
     */
    void HandleBurstTx(FragmentsInBurst& burstMap,
                       Ptr<const Packet> burst,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the burst reception traceback.
     *
     * \param burstMap the burst map indicates contained fragments
     * \param map the in-flight packet map
     * \param burst the received burst
     * \param header the header of the created burst
     */
    void HandleBurstRx(FragmentsInBurst& burstMap,
                       InFlightPacketMap& map,
                       Ptr<const Packet> burst,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation creation traceback.
     *
     * \param map the in-flight packet map
     * \param nodeId the node ID
     * \param fragment the created fragment
     * \param dstPort the destination port
     * \param header the header of the created fragment
     */
    void HandleFragmentTx(InFlightPacketMap& map,
                          uint32_t nodeId,
                          Ptr<const Packet> fragment,
                          uint16_t dstPort,
                          const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation reception traceback.
     *
     * \param map the in-flight packet map
     * \param fragment the received fragment
     */
    void HandleFragmentRx(InFlightPacketMap& map, Ptr<const Packet> fragment);

  private:
    std::map<uint32_t /* Node ID */, std::vector<std::pair<Time, double>>> m_e2eDelay;
};

} // namespace ns3

#endif /* ENV_APP_TRACER_H */