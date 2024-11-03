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
     * Handles frames comes from WifiMac::m_ackedMpduCallback.
     */
    void AckedMpdu(Ptr<const WifiMpdu> mpdu);

  protected:
    void DoDispose() override;

    /**
     * Trace the packet that has been received by the WifiNetDevice and is about to be enqueued.
     *
     * \param packet the packet to be enqueued.
     */
    void MacTx(Ptr<const Packet> packet);

    /**
     * Trace the packet that has been received by the MAC and is about to be forwarded up to the
     * WifiNetDevice.
     *
     * \param packet the packet to be forwarded up.
     */
    void MacRx(Ptr<const Packet> packet);

  private:
    Ptr<WifiMac> m_mac;
    Time m_duration;
};

} // namespace ns3

#endif /* SENSING_MAC_H */
