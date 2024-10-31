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

    void Sensing(Mac48Address sender, double rssi, Time duration);

  protected:
  private:
};

} // namespace ns3

#endif /* SENSING_MAC_H */
