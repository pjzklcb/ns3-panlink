/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_CHANNEL_ACCESS_MANAGER_H
#define PANLINK_CHANNEL_ACCESS_MANAGER_H

#include "ns3/channel-access-manager.h"

namespace ns3
{

class PanlinkTxop;

/**
 * \ingroup panlink
 */
class PanlinkChannelAccessManager : public ChannelAccessManager
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkChannelAccessManager();
    ~PanlinkChannelAccessManager() override;

  protected:
  private:
    /**
     * Restore the deferred access if needed.
     */
    void DoRestoreAccessIfNeeded();

    /**
     * Called when access restored from deferring
     * (e.g. backoff procedure restart).
     */
    void AccessRestore();

    EventId m_accessRestore;  //!< the access restore event
};

} // namespace ns3

#endif /* PANLINK_CHANNEL_ACCESS_MANAGER_H */
