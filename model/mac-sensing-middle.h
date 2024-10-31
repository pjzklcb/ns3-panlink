/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef MAC_SENSING_MIDDLE_H
#define MAC_SENSING_MIDDLE_H

#include "ns3/simple-ref-count.h"
#include "ns3/mac48-address.h"
#include "ns3/nstime.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class MacSensingMiddle : public SimpleRefCount<MacSensingMiddle>
{
  public:
    MacSensingMiddle();
    ~MacSensingMiddle();

    void Sensing(Mac48Address sender, double rssi, Time duration);

  protected:

  private:

};

} // namespace ns3

#endif /* MAC_SENSING_MIDDLE_H */
