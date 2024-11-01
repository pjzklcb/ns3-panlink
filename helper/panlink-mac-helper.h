/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_MAC_HELPER_H
#define PANLINK_MAC_HELPER_H

#include "ns3/wifi-mac-helper.h"

namespace ns3
{

/**
 * \brief create MAC layers for a ns3::WifiNetDevice.
 *
 * This class can create MACs of type ns3::ApWifiMac, ns3::StaWifiMac and ns3::AdhocWifiMac.
 * Its purpose is to allow a WifiHelper to configure and install WifiMac objects on a collection
 * of nodes. The WifiMac objects themselves are mainly composed of TxMiddle, RxMiddle,
 * ChannelAccessManager, FrameExchangeManager, WifiRemoteStationManager, MpduAggregator and
 * MsduAggregartor objects, so this helper offers the opportunity to configure attribute values away
 * from their default values, on a per-NodeContainer basis. By default, it creates an Adhoc MAC
 * layer without QoS. Typically, it is used to set type and attribute values, then hand this object
 * over to the WifiHelper that finishes the job of installing.
 */
class PanlinkMacHelper : public WifiMacHelper
{
  public:
    PanlinkMacHelper();
    ~PanlinkMacHelper() override;

    Ptr<WifiMac> Create(Ptr<WifiNetDevice> device, WifiStandard standard) const override;

    /**
     * Helper function used to set the Frame Exchange Manager object.
     *
     * \tparam Args \deduced Template type parameter pack for the sequence of name-value pairs.
     * \param args A sequence of name-value pairs of the attributes to set.
     */
    template <typename... Args>
    void SetFrameExchangeManager(std::string type, Args&&... args);

  protected:
    ObjectFactory m_sensingMac; ///< sensing MAC object factory
};

} // namespace ns3

/***************************************************************
 *  Implementation of the templates declared above.
 ***************************************************************/

namespace ns3
{

template <typename... Args>
void
PanlinkMacHelper::SetFrameExchangeManager(std::string type, Args&&... args)
{
    m_frameExchangeManager.SetTypeId(type);
    m_frameExchangeManager.Set(args...);
}

} // namespace ns3

#endif /* PANLINK_MAC_HELPER_H */
