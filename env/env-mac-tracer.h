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

#include "ns3/object.h"
#include "ns3/nstime.h"
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
     * \param item the enqueued packet
     */
    void HandleEdcaEnqueue(Ptr<const WifiMpdu> item);
    /**
     * Handle the mpdu dequeue traceback.
     * 
     * \param item the dequeued packet
     */
    void HandleEdcaDequeue(Ptr<const WifiMpdu> item);
    /**
     * Handle the mpdu expired traceback.
     * 
     * \param item the expired packet
     */
    void HandleEdcaExpired(Ptr<const WifiMpdu> item);

    /**
     * Handle the mac forwardup traceback.
     * 
     * \param p the forwarded packet
     */
    void HandleMacForwardUp(Ptr<const Packet> p);

    /**
     * Handle the mpdu acked traceback.
     * 
     * \param context string contains the node ID
     * \param item the acked packet
     */
    void HandleMpduAcked(std::string context, Ptr<const WifiMpdu> item);

    /**
     * Handle contention window update traceback.
     * 
     * \param context string contains the node ID
     * \param cw the updated contention window
     * \param linkId the link ID
     */
    void HandleCwUpdate(std::string context, uint32_t cw, uint8_t linkId);
    /**
     * Handle the new backoff traceback.
     * 
     * \param context string contains the node ID
     * \param backoff the new backoff
     * \param linkId the link ID
     */
    void HandleNewBackoff(std::string context, uint32_t backoff, uint8_t linkId);

  private:
    std::vector<std::pair<Time, uint32_t>> m_cw;
    std::vector<std::pair<Time, uint32_t>> m_backoff;
};

} // namespace ns3

#endif /* ENV_MAC_TRACER_H */