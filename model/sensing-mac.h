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
     * Collect frame information.
     */
    void Sensing(Mac48Address sender, double rssi, Time duration);

  protected:
  private:
    Ptr<WifiMac> m_mac;
    uint8_t m_linkId;
};

} // namespace ns3

#endif /* SENSING_MAC_H */
