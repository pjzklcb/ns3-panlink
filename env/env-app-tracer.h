/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_APP_TRACER_H
#define ENV_APP_TRACER_H

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
     * \param nodeId the node ID
     * \param packet the created packet
     * \param dstPort the destination port
     */
    void HandleAppTx(uint32_t nodeId, Ptr<const Packet> packet, uint16_t dstPort);
    /**
     * Handle the packet reception traceback.
     *
     * \param nodeId the node ID
     * \param packet the received packet
     * \param dstPort the destination port
     */
    void HandleAppRx(uint32_t nodeId, Ptr<const Packet> packet, uint16_t dstPort);

    /**
     * Handle the burst creation traceback.
     *
     * \param nodeId the node ID
     * \param burst the created burst
     * \param dstPort the destination port
     * \param header the header of the created burst
     */
    void HandleBurstTx(uint32_t nodeId,
                       Ptr<const Packet> burst,
                       uint16_t dstPort,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the burst reception traceback.
     *
     * \param nodeId the node ID
     * \param burst the received burst
     * \param dstPort the destination port
     * \param header the header of the created burst
     */
    void HandleBurstRx(uint32_t nodeId,
                       Ptr<const Packet> burst,
                       uint16_t dstPort,
                       const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation creation traceback.
     *
     * \param nodeId the node ID
     * \param fragment the created fragment
     * \param dstPort the destination port
     * \param header the header of the created fragment
     */
    void HandleFragmentTx(uint32_t nodeId,
                          Ptr<const Packet> fragment,
                          uint16_t dstPort,
                          const SeqTsSizeFragHeader& header);
    /**
     * Handle the fragmentation reception traceback.
     *
     * \param nodeId the node ID
     * \param fragment the received fragment
     * \param dstPort the destination port
     * \param header the header of the created fragment
     */
    void HandleFragmentRx(uint32_t nodeId,
                          Ptr<const Packet> fragment,
                          uint16_t dstPort,
                          const SeqTsSizeFragHeader& header);

  private:
    std::map<uint32_t /* Node ID */, std::vector<std::pair<Time, double>>> m_e2eDelay;
};

} // namespace ns3

#endif /* ENV_APP_TRACER_H */