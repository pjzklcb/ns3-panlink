/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef SENSING_MAC_H
#define SENSING_MAC_H

#include "ns3/mac48-address.h"
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/packet.h"
#include "ns3/wifi-mac.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class SensingMac : public Object
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    SensingMac();
    ~SensingMac();

    /**
     * Set the MAC layer to use.
     *
     * \param mac the MAC layer to use
     */
    virtual void SetWifiMac(const Ptr<WifiMac> mac);
    /**
     * Set the maximum duration of the data stored.
     *
     * \param duration the duration after which the data is considered stale
     */
    void SetDuration(Time duration);

    /**
     * Handles frames comes from FrameExchangeManager::Sensing.
     *
     * \param sender the MAC address of the sender
     * \param rssi the RSSI of the frame
     * \param duration the duration of the frame
     * \param linkId the ID of the link
     */
    void FrameExchangeSensing(Mac48Address sender, double rssi, Time duration, uint8_t linkId);

    /**
     * Handles frames comes from PanlinkWifiMac::Sensing.
     *
     * \param mpdu the MPDU
     * \param linkId the ID of the link
     */
    void WifiMacSensing(Ptr<const WifiMpdu> mpdu, uint8_t linkId);

    /**
     * Trace the packet that has been enqueued to the given AC MAC queue.
     *
     * \param ac the AC of the MacQueue
     * \param item the enqueued MPDU
     */
    void NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item);

    /**
     * Trace the packet that has been sent by wifi PHY.
     *
     * \param context the string contains link ID
     * \param packet the sent packet
     * \param txPowerW the transmission power
     */
    void NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW);

    /**
     * Handles frames comes from WifiMac::m_ackedMpduCallback.
     *
     * \param mpdu the MPDU
     */
    void AckedMpdu(Ptr<const WifiMpdu> mpdu);

    struct SelfPacketInfo
    {
        uint64_t m_uid;
        uint32_t m_packetSize;
        AcIndex m_ac;
        uint8_t m_linkId;
        Time m_enqueueTime;
        Time m_duration;
        Time m_ackedTime{Seconds(0)};
        std::vector<Time> m_phyTxTime;
        double m_txPowerW;
    };

  protected:
    void DoDispose() override;

    /**
     * Wipe all expired packets that is considered stale.
     */
    virtual void WipeAllExpiredPackets();

  private:
    Ptr<WifiMac> m_mac;
    Time m_duration; ///< maximum duration of the data stored
    std::map<uint64_t /* UID */, SelfPacketInfo> m_selfPacketMap;
};

} // namespace ns3

#endif /* SENSING_MAC_H */
