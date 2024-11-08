/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_STA_WIFI_MAC_H
#define PANLINK_STA_WIFI_MAC_H

#include "ns3/sta-wifi-mac.h"


namespace ns3
{

/**
 * \ingroup panlink
 */
class PanlinkStaWifiMac : public StaWifiMac
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkStaWifiMac();
    ~PanlinkStaWifiMac() override;

  protected:

  private:

};

} // namespace ns3

#endif /* PANLINK_STA_WIFI_MAC_H */
