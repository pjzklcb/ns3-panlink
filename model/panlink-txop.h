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

    void NotifyChannelReleased(uint8_t linkId) override;

    /**
     * Get the backoff value of the given link.
     *
     * @param linkId the link ID
     */
    uint32_t GetBackoffSlots(uint8_t linkId) const;

    /**
     * Get the times of deferred access on the given link by far.
     *
     * @param linkId the link ID
     */
    uint32_t GetNDeferredAccess(uint8_t linkId) const;

    /**
     * Get whether panlink channel access strategy is supported by MAC.
     */
    bool GetEnablePanlinkChannelAccess() const;

  protected:
    ///< ChannelAccessManager associated class
    friend class ChannelAccessManager;
    friend class PanlinkChannelAccessManager;
    friend class ::EmlsrUlTxopTest;

    void DoInitialize() override;
    void GenerateBackoff(uint8_t linkId) override;
    void UpdateBackoffSlotsNow(uint32_t nSlots, Time backoffUpdateBound, uint8_t linkId) override;

    /**
     * Generate new backoff for the given link after deferred access.
     *
     * @param linkId the link ID
     */
    void GenerateNewBackoff(uint8_t linkId);

    /**
     * Reset access related parameters when channel released.
     * 
     * @param linkId the link ID
     */
    void ResetBackoff(uint8_t linkId);

    /**
     * Structure holding information specific to a single link. Here, the meaning of
     * "link" is that of the 11be amendment which introduced multi-link devices. For
     * previous amendments, only one link can be created.
     */
    struct PanLinkEntity : public QosTxop::QosLinkEntity
    {
        /// Destructor (a virtual method is needed to make this struct polymorphic)
        ~PanLinkEntity() override = default;

        uint32_t waitedSlots{0};     //!< total waited slots
        uint32_t nDeferredAccess{0}; //!< deferred access times
    };

    /**
     * Get a reference to the link associated with the given ID.
     *
     * @param linkId the given link ID
     * \return a reference to the link associated with the given ID
     */
    PanLinkEntity& GetLink(uint8_t linkId) const;

  private:
    std::unique_ptr<LinkEntity> CreateLinkEntity() const override;

    /// TracedCallback for waited slots trace value typedef
    typedef TracedCallback<uint32_t /* deferred times */,
                           uint32_t /* waited slots */,
                           uint32_t /* backoff */,
                           uint8_t /* linkId */>
        WaitedSlotsValueTracedCallback;

    WaitedSlotsValueTracedCallback m_waitedSlotsTrace; //!< waited slots trace value
};

} // namespace ns3

#endif /* PANLINK_TXOP_H */
