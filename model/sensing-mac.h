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
     * Set the ID of the link this Frame Exchange Manager is associated with.
     *
     * \param linkId the ID of the link this Frame Exchange Manager is associated with
     */
    virtual void SetLinkId(uint8_t linkId);
    /**
     * Set the MAC layer to use.
     *
     * \param mac the MAC layer to use
     */
    virtual void SetWifiMac(const Ptr<WifiMac> mac);

    /**
     * Handles frames comes from FrameExchangeManager::Sensing.
     * 
     * \param sender the MAC address of the sender
     * \param rssi the RSSI of the frame
     * \param duration the duration of the frame
     */
    void FrameExchangeSensing(Mac48Address sender, double rssi, Time duration);

    /**
     * Handles frames comes from PanlinkWifiMac::Sensing.
     * 
     */
    void WifiMacSensing(Ptr<const WifiMpdu> mpdu, uint8_t linkId);

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
    uint8_t m_linkId;
};

} // namespace ns3

#endif /* SENSING_MAC_H */
