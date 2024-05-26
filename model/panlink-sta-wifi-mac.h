/*
 * Copyright (c) 2024
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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

} // namespace ns3

#endif /* PANLINK_STA_WIFI_MAC_H */
