/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_MONITOR_H
#define ENV_MONITOR_H

#include "env-utils.h"
#include "env-app-tracer.h"
#include "env-mac-tracer.h"
#include "env-phy-tracer.h"

#include "ns3/panlink-utils.h"
#include "ns3/singleton.h"
#include "ns3/vr-app-module.h"
#include "ns3/wifi-module.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvMonitor : public Singleton<EnvMonitor>
{
  public:
    /**
     * Connect all the traceback sources.
     */
    void ConnectTraceSources();

  private:
    /**
     * Bind trace sources related to APP.
     */
    void ConnectAppTrace();
    /**
     * Bind trace sources related to MAC.
     */
    void ConnectMacTrace();
    /**
     * Bind trace sources related to PHY.
     */
    void ConnectPhyTrace();

    /**
     * Notify that a packet has been created by the app.
     *
     * \param context string contains the node ID
     * \param packet the created packet
     * \param from source inet address
     * \param to destination inet address
     */
    void NotifyAppTx(std::string context,
                     Ptr<const Packet> packet,
                     const Address& from,
                     const Address& to);
    /**
     * Notify that a packet has been received by the app.
     */
    void NotifyAppRx(std::string context,
                     Ptr<const Packet> packet,
                     const Address& from,
                     const Address& to);
    /**
     * Notify that a packet has been created by the burst app.
     *
     * \param context string contains the node ID
     * \param burst the created burst
     * \param from source inet address
     * \param to destination inet address
     * \param header the header of the created burst
     */
    void NotifyBurstTx(std::string context,
                       Ptr<const Packet> burst,
                       const Address& from,
                       const Address& to,
                       const SeqTsSizeFragHeader& header);

    /**
     * Notify that a packet has been received by the burst app.
     */
    void NotifyBurstRx(std::string context,
                       Ptr<const Packet> burst,
                       const Address& from,
                       const Address& to,
                       const SeqTsSizeFragHeader& header);
    /**
     * Notify that a fragment of a burst has been created.
     */
    void NotifyFragmentTx(std::string context,
                          Ptr<const Packet> fragment,
                          const Address& from,
                          const Address& to,
                          const SeqTsSizeFragHeader& header);
    /**
     * Notify that a fragment of a burst has been received.
     */
    void NotifyFragmentRx(std::string context,
                          Ptr<const Packet> fragment,
                          const Address& from,
                          const Address& to,
                          const SeqTsSizeFragHeader& header);

    /**
     * Notify that a packet has been enqueued into an EDCA queue.
     * 
     * \param item the enqueued packet
     */
    void NotifyEdcaEnqueue(Ptr<const WifiMpdu> item);
    /**
     * Notify that a packet has been dequeued from an EDCA queue.
     */
    void NotifyEdcaDequeue(Ptr<const WifiMpdu> item);
    /**
     * Notify that a packet has been expired.
     */
    void NotifyEdcaExpired(Ptr<const WifiMpdu> item);

    /**
     * Notify that a packet has been forwarded up by the Mac Layer.
     * 
     * \param p the forwarded packet
     */
    void NotifyMacForwardUp(Ptr<const Packet> p);

    /**
     * Notify that a packet has been acked.
     * 
     * \param context string contains the node ID
     * \param item the acked packet
     */
    void NotifyMpduAcked(std::string context, Ptr<const WifiMpdu> item);

    /**
     * Notify that CW has been updated.
     * 
     * \param context string contains the node ID
     * \param cw the updated contention window
     * \param linkId the link ID
     */
    void NotifyCwUpdate(std::string context, uint32_t cw, uint8_t linkId);
    /**
     * Notify that new backoff has been created.
     * 
     * \param context string contains the node ID
     * \param backoff the new backoff
     * \param linkId the link ID
     */
    void NotifyNewBackoff(std::string context, uint32_t backoff, uint8_t linkId);

    /**
     * Notify that a packet is start to be transmitted.
     * 
     * \param context string contains the node ID
     * \param p the packet
     * \param txPowerW the tx power
     */
    void NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW);
    /**
     * Notify that the transmission of a packet is ended.
     * 
     * \param context string contains the node ID
     * \param p the packet
     */
    void NotifyPhyTxEnd(std::string context, Ptr<const Packet> p);

    NodeContainer m_nodes;
    NetDeviceContainer m_devices;
    Ipv4InterfaceContainer m_ipv4Interfaces;
    SrcAppMap m_srcApps;

    Ptr<EnvAppTracer> m_appTracer;
    Ptr<EnvMacTracer> m_macTracer;
    Ptr<EnvPhyTracer> m_phyTracer;

    std::map<Mac48Address, uint32_t /* Node ID */> m_macAddressMap;
    std::map<Ipv4Address, uint32_t /* Node ID */> m_ipv4AddressMap;

    InFlightPacketMap m_packetMap;
    InFlightBurstMap m_burstMap;
};

} // namespace ns3

#endif /* ENV_MONITOR_H */
