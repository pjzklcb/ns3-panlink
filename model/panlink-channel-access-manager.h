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

    bool NeedBackoffUponAccess(Ptr<Txop> txop, bool hadFramesToTransmit, bool checkMediumBusy) override;

  protected:
    void DisableEdcaFor(Ptr<Txop> qosTxop, Time duration) override;
    void NotifyRxStartNow(Time duration) override;
    void NotifyTxStartNow(Time duration) override;
    void NotifyCcaBusyStartNow(Time duration,
                               WifiChannelListType channelType,
                               const std::vector<Time>& per20MhzDurations) override;
    void NotifyNavResetNow(Time duration) override;
    void NotifyNavStartNow(Time duration) override;

  private:
    /**
     * Resume the deferred access if needed.
     */
    void DoResumeAccessIfNeeded();

    /**
     * Called when access resumed from deferring
     * (e.g. backoff procedure restart).
     */
    void AccessResume();

    EventId m_accessResume; //!< the access resume event
};

} // namespace ns3

#endif /* PANLINK_CHANNEL_ACCESS_MANAGER_H */
