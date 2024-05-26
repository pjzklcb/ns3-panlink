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

#include "panlink-sta-wifi-mac.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("PanlinkStaWifiMac");

NS_OBJECT_ENSURE_REGISTERED(PanlinkStaWifiMac);

TypeId
PanlinkStaWifiMac::GetTypeId()
{
    static TypeId tid =
        TypeId("ns3::PanlinkStaWifiMac")
            .SetParent<StaWifiMac>()
            .SetGroupName("Panlink")
            .AddConstructor<PanlinkStaWifiMac>();
    return tid;
}

PanlinkStaWifiMac::PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

PanlinkStaWifiMac::~PanlinkStaWifiMac()
{
    NS_LOG_FUNCTION(this);
}

} // namespace ns3
