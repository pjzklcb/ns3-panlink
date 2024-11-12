/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_TXOP_H
#define PANLINK_TXOP_H

#include "ns3/qos-txop.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class PanlinkTxop : public QosTxop
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkTxop();
    ~PanlinkTxop() override;

  protected:
  private:
};

} // namespace ns3

#endif /* PANLINK_TXOP_H */
