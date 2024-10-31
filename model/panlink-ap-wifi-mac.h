/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_AP_WIFI_MAC_H
#define PANLINK_AP_WIFI_MAC_H

#include "ns3/ap-wifi-mac.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class PanlinkApWifiMac : public ApWifiMac
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkApWifiMac();
    ~PanlinkApWifiMac() override;

  protected:

  private:
};

} // namespace ns3

#endif /* PANLINK_AP_WIFI_MAC_H */
