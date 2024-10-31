/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_STA_WIFI_MAC_H
#define PANLINK_STA_WIFI_MAC_H

#include "sensing-mac.h"

#include "ns3/sta-wifi-mac.h"

namespace ns3
{

class FrameExchangeManager;

/**
 * \ingroup panlink
 */
class PanlinkStaWifiMac : public StaWifiMac
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    PanlinkStaWifiMac();
    ~PanlinkStaWifiMac() override;

  protected:
    /**
     * Structure holding information specific to a single link. Here, the meaning of
     * "link" is that of the 11be amendment which introduced multi-link devices. For
     * previous amendments, only one link can be created.
     */
    struct PanStaLinkEntity : public StaWifiMac::StaLinkEntity
    {
        /// Destructor (a virtual method is needed to make this struct polymorphic)
        ~PanStaLinkEntity() override;

        Ptr<SensingMac> sensingMac; // sensing MAC object
    };

    /**
     * Get a reference to the link associated with the given ID.
     *
     * \param linkId the given link ID
     * \return a reference to the link associated with the given ID
     */
    PanStaLinkEntity& GetLink(uint8_t linkId) const;

    /**
     * Cast the given LinkEntity object to PanStaLinkEntity.
     *
     * \param link the given LinkEntity object
     * \return a reference to the object casted to PanStaLinkEntity
     */
    PanStaLinkEntity& GetPanStaLink(const std::unique_ptr<WifiMac::LinkEntity>& link) const;

  private:
    std::unique_ptr<LinkEntity> CreateLinkEntity() const override;

    Ptr<SensingMac> m_sensingMac;
};

} // namespace ns3

#endif /* PANLINK_STA_WIFI_MAC_H */
