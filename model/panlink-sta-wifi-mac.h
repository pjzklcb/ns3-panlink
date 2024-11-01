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

    /**
     * \param sensingMacs the sensing MACs attached to this MAC.
     */
    void SetSensingMacs(const std::vector<Ptr<SensingMac>>& sensingMacs);

    /**
     * Get the Sensing MAC associated with the given link
     *
     * \param linkId the ID of the given link
     * \return the Sensing MAC
     */
    Ptr<SensingMac> GetSensingMac(uint8_t linkId = SINGLE_LINK_OP_ID) const;

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

    /**
     * Handles all the management and data frames received by the StaWifiMac::Receive callback on
     * the given link. Only these frames are forwarded up from MacRxMiddle.
     *
     * \param mpdu the MPDU that has been received.
     * \param linkId the ID of the given link
     */
    virtual void Sensing(Ptr<const WifiMpdu> mpdu, uint8_t linkId);

  private:
    std::unique_ptr<LinkEntity> CreateLinkEntity() const override;
};

} // namespace ns3

#endif /* PANLINK_STA_WIFI_MAC_H */
