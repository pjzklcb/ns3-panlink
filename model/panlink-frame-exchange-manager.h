/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_FRAME_EXCHANGE_MANAGER_H
#define PANLINK_FRAME_EXCHANGE_MANAGER_H

#include "sensing-mac.h"

#include "ns3/he-frame-exchange-manager.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class PanlinkFrameExchangeManager : public HeFrameExchangeManager
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkFrameExchangeManager();
    ~PanlinkFrameExchangeManager() override;

    void SetWifiPhy(const Ptr<WifiPhy> phy) override;

    /**
     * Set sensing MAC.
     * 
     * \param sensingMac sensing MAC
     */
    virtual void SetSensingMac(const Ptr<SensingMac> sensingMac);

  protected:
    void Sensing(Ptr<const WifiPsdu> psdu,
                 RxSignalInfo rxSignalInfo,
                 const WifiTxVector& txVector,
                 const std::vector<bool>& perMpduStatus);

  private:
    Ptr<SensingMac> m_sensingMac;
};

} // namespace ns3

#endif /* PANLINK_FRAME_EXCHANGE_MANAGER_H */
