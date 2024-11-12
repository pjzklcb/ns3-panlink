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
     * Schedule the deferred access end event.
     */
    void DoScheduleDeferredAccessEnd();

    EventId m_deferredAccessEnd;  //!< the deferred access end event
};

} // namespace ns3

#endif /* PANLINK_CHANNEL_ACCESS_MANAGER_H */
