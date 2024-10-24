/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_MAC_TRACER_H
#define ENV_MAC_TRACER_H

#include "env-utils.h"

#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/wifi-mpdu.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvMacTracer : public Object
{
  public:
    /**
     * Handle the mpdu enqueue traceback.
     *
     * \param map the in-flight packet map
     * \param nodeId the node ID
     * \param item the enqueued packet
     * \param ac the AC
     */
    void HandleEdcaEnqueue(InFlightPacketMap& map,
                           uint32_t nodeId,
                           Ptr<const WifiMpdu> item,
                           AcIndex ac);
    /**
     * Handle the mpdu dequeue traceback.
     *
     * \param map the in-flight packet map
     * \param item the dequeued packet
     */
    void HandleEdcaDequeue(InFlightPacketMap& map, Ptr<const WifiMpdu> item);
    /**
     * Handle the mpdu expired traceback.
     *
     * \param map the in-flight packet map
     * \param item the expired packet
     */
    void HandleEdcaExpired(InFlightPacketMap& map, Ptr<const WifiMpdu> item);

    /**
     * Handle the mac forwardup traceback.
     *
     * \param map the in-flight packet map
     * \param p the forwarded packet
     */
    void HandleMacForwardUp(InFlightPacketMap& map, Ptr<const Packet> p);

    /**
     * Handle the acked mpdu traceback.
     *
     * \param map the in-flight packet map
     * \param item the acked packet
     */
    void HandleAckedMpdu(InFlightPacketMap& map, Ptr<const WifiMpdu> item);

    /**
     * Handle contention window update traceback.
     *
     * \param nodeId the node ID
     * \param cw the updated contention window
     * \param linkId the link ID
     * \param ac the AC
     */
    void HandleCwUpdate(uint32_t nodeId, uint32_t cw, uint8_t linkId, AcIndex ac);
    /**
     * Handle the new backoff traceback.
     *
     * \param nodeId the node ID
     * \param backoff the new backoff
     * \param linkId the link ID
     * \param ac the AC
     */
    void HandleNewBackoff(uint32_t nodeId, uint32_t backoff, uint8_t linkId, AcIndex ac);

    /**
     * Save the CW to CSV file.
     *
     * \param resultDir the directory path
     */
    void WriteCwToCsv(std::string resultDir);
    /**
     * Save the backoff to CSV file.
     *
     * \param resultDir the directory path
     */
    void WriteBackoffToCsv(std::string resultDir);

  private:
    typedef std::map<AcIndex, std::vector<std::pair<Time, uint32_t>>> EdcaQueueMap;

    std::map<uint32_t /* Node ID */, EdcaQueueMap> m_cw;
    std::map<uint32_t /* Node ID */, EdcaQueueMap> m_backoff;
};

} // namespace ns3

#endif /* ENV_MAC_TRACER_H */