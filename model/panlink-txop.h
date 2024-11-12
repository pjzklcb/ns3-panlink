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
    /**
     * Structure holding information specific to a single link. Here, the meaning of
     * "link" is that of the 11be amendment which introduced multi-link devices. For
     * previous amendments, only one link can be created.
     */
    struct PanLinkEntity : public QosTxop::QosLinkEntity
    {
        /// Destructor (a virtual method is needed to make this struct polymorphic)
        ~PanLinkEntity() override = default;

        uint32_t waitedSlots{0};  //!< total waited slots
        uint32_t nAccessDefer{0}; //!< access defer times
    };

    /**
     * Get a reference to the link associated with the given ID.
     *
     * \param linkId the given link ID
     * \return a reference to the link associated with the given ID
     */
    PanLinkEntity& GetLink(uint8_t linkId) const;

  private:
    std::unique_ptr<LinkEntity> CreateLinkEntity() const override;
};

} // namespace ns3

#endif /* PANLINK_TXOP_H */
