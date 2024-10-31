/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#include "mac-sensing-middle.h"

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("MacSensingMiddle");

MacSensingMiddle::MacSensingMiddle()
{
    NS_LOG_FUNCTION(this);
}

MacSensingMiddle::~MacSensingMiddle()
{
    NS_LOG_FUNCTION(this);
}

void 
MacSensingMiddle::Sensing(Mac48Address sender, double rssi, Time duration)
{
    NS_LOG_FUNCTION(this << sender << rssi << duration);
}

} // namespace ns3
