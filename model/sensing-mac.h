/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef SENSING_MAC_H
#define SENSING_MAC_H

#include "panlink-sta-wifi-mac.h"

#include "ns3/mac48-address.h"
#include "ns3/nstime.h"
#include "ns3/object.h"
#include "ns3/packet.h"

#include <deque>

namespace ns3
{

/**
 * \ingroup panlink
 */
class SensingMac : public Object
{
  public:
    /**
     * \brief Get the type ID.
     * @returns the object TypeId
     */
    static TypeId GetTypeId();

    SensingMac();
    ~SensingMac();

    /**
     * Set the MAC layer to use.
     *
     * @param mac the MAC layer to use
     */
    virtual void SetWifiMac(const Ptr<WifiMac> mac);
    /**
     * Set the maximum duration of the data stored.
     *
     * @param duration the duration after which the data is considered stale
     */
    void SetMaxDuration(Time duration);

    /**
     * The information of the packets sent by this STA.
     */
    struct SelfPacketInfo
    {
        uint32_t packetSize;
        AcIndex ac;
        uint8_t linkId;
        Time enqueueTime;
        Time duration;
        Time ackedTime{Seconds(0)};
        uint32_t txCounter{0};
        bool isExpired{false};
        bool isDropped{false};
        double txPowerW;
    };

    /**
     * The channel utilization of this STA on the given link.
     */
    struct SelfLinkInfo
    {
        Time lastTxOkTime;
        std::list<uint64_t> pendingPackets; //!< pending packets
        std::map<uint64_t /* UID */, std::pair<Time /* Tx Begin */, Time /* Tx End */>> channelUtil;
    };

    /**
     * The information of the queues of each AC.
     */
    struct MacQueueInfo
    {
        std::list<uint64_t> expiredPackets; // list of expired packets
        std::list<uint64_t> droppedPackets; // list of dropped packets
    };

    /**
     * Trace the packet that has been enqueued to the given AC MAC queue.
     *
     * @param ac the AC of the MacQueue
     * @param item the enqueued MPDU
     */
    void NotifyEdcaEnqueue(std::string ac, Ptr<const WifiMpdu> item);
    /**
     * Trace the packet that is expired in the given AC MAC queue.
     *
     * @param ac the AC of the MacQueue
     * @param item the expired MPDU
     */
    void NotifyEdcaExpired(std::string ac, Ptr<const WifiMpdu> item);

    /**
     * Trace the packet that has been sent by wifi PHY.
     *
     * @param context the string contains link ID
     * @param packet the sent packet
     * @param txPowerW the transmission power
     */
    void NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW);
    /**
     * Trace the packet that finished transmission by wifi PHY.
     *
     * @param context the string contains link ID
     * @param packet the sent packet
     * @param txPowerW the transmission power
     */
    void NotifyPhyTxEnd(std::string context, Ptr<const Packet> p);

    /**
     * Handles frames comes from WifiMac::m_ackedMpduCallback.
     *
     * @param mpdu the MPDU
     */
    void NotifyAckedMpdu(Ptr<const WifiMpdu> mpdu);
    /**
     * Handles frames comes from WifiMac::m_droppedMpduCallback.
     *
     * @param reason the dropped reason
     * @param mpdu the MPDU
     */
    void NotifyDroppedMpdu(WifiMacDropReason reason, Ptr<const WifiMpdu> mpdu);

    /**
     * @param linkId the link ID
     * @returns the last successful transmission time of this STA on the given link
     */
    Time GetSelfLastTxOk(uint8_t linkId) const;

    /**
     * Get the channel utilization of previous period of this STA on the given link.
     *
     * @param period the period of time
     * @param linkId the link ID
     */
    double GetSelfChannelUtil(Time period, uint8_t linkId) const;

    /**
     * Get the required throughput for the given ac and link ID in the past period of time.
     *
     * @param period the period
     * @param ac the AC
     * @param linkId the ID of the link
     * @returns the throughput in Mbps
     */
    double GetThptReq(Time period, AcIndex ac, uint8_t linkId = SINGLE_LINK_OP_ID) const;

    /**
     * Get the actual throughput of the given ac and link ID in the past period of time.
     *
     * @param period the period
     * @param ac the AC
     * @param linkId the ID of the link
     * @returns the throughput in Mbps
     */
    double GetThpt(Time period, AcIndex ac, uint8_t linkId = SINGLE_LINK_OP_ID) const;

    /**
     * Get the average delay of the given ac and link ID in the past period of time.
     *
     * @param period the period
     * @param ac the AC
     * @param linkId the ID of the link
     * @returns the average delay in milliseconds (0 if no packet tx successfully)
     */
    double GetDelay(Time period, AcIndex ac, uint8_t linkId = SINGLE_LINK_OP_ID) const;

    /**
     * The role for neighboring STAs.
     */
    enum NeighborStaRole
    {
        ASSOCIATED_AP = 0,
        BSS_STA,
        OBSS,
        UNKNOWN
    };

    /**
     * Information of the neighbor STAs.
     */
    struct NeighborStaInfo
    {
        NeighborStaRole role{UNKNOWN};
        uint8_t linkId;
        Time lastTxOkTime{Seconds(0)};   // the time of the last successful transmission
        Time lastTxDataTime{Seconds(0)};     // the time of the last transmission
        std::deque<Time> txDataIntervals;
        uint32_t nTxPackets{0};
        uint32_t nTxDataPackets{0};
        uint32_t nAckedPackets{0};
        std::map<Time /* Rx End */, Time /* duration */> channelUtil;
        double rssi;
    };

    /**
     * Handles received frames not matter the STA they are sent to.
     *
     * @param hdr the MAC header of the frame
     * @param rssi the RSSI of the frame
     * @param linkId the ID of the link
     */
    void NotifyFrameInfo(const WifiMacHeader& hdr, double rssi, uint8_t linkId);

    /**
     * Record the STA whose last transmission was successful.
     *
     * @param sta the MAC address of the STA
     * @param linkId the ID of the link
     */
    void NotifyLastTxOkSta(Mac48Address sta, uint8_t linkId);

    /**
     * Get the minimum RSSI of BSS STAs on the given link.
     *
     * @param linkId the link ID
     */
    double GetMinBssRssi(uint8_t linkId) const;
    /**
     * Get the maximum RSSI of OBSS STAs on the given link.
     *
     * @param linkId the link ID
     */
    double GetMaxObssRssi(uint8_t linkId) const;

    /**
     * @param delay the threshold delay
     * @param linkId the link ID
     * @returns the number of STAs whose last successful transmission was less than the given delay
     */
    uint32_t GetNStaTxOkDelayLessThan(Time delay, uint8_t linkId) const;
    /**
     * @param interval the threshold interval
     * @param linkId the link ID
     * @returns the expectation of data intervals of neighbor STAs that are less than the given interval
     */
    double GetNTxDataIntervalLessThan(Time interval, uint8_t linkId) const;

    /**
     * @param linkId the link ID
     * @returns the number of STAs with non-empty MAC queue at the moment.
     */
    uint32_t GetNStasWithPacketsNow(uint8_t linkId);
    /**
     * @param linkId the link ID
     * @returns the minimum backoff of neighbor STAs
     */
    uint32_t GetMinNeighborBackoff(uint8_t linkId);

    /**
     * Get the number of STAs in the same BSS on the given link.
     *
     * @param linkId the link ID
     */
    uint32_t GetNBssStas(uint8_t linkId) const;
    /**
     * Get the number of STAs belongs to OBSS on the given link.
     *
     * @param linkId the link ID
     */
    uint32_t GetNObssStas(uint8_t linkId) const;

    /**
     * Get the channel utilization of previous period of STAs within the same BSS on the given link.
     *
     * @param period the period of time
     * @param linkId the link ID
     */
    double GetBssChannelUtil(Time period, uint8_t linkId) const;
    /**
     * Get the channel utilization of previous period of other STAs on the given link.
     *
     * @param period the period of time
     * @param linkId the link ID
     */
    double GetOtherChannelUtil(Time period, uint8_t linkId) const;

    /**
     * The obs collected from the mac since the last access restore.
     */
    struct AccessInfo
    {
        uint32_t nInQueuePackets{0};
        uint32_t nExpiredPackets{0};
        uint32_t nDroppedPackets{0};
        uint32_t nSendingPackets{0};
        uint32_t nAckedPackets{0};
        uint32_t nLowDelayPackets{0};  // latency <= 10ms
        uint32_t nHighDelayPackets{0}; // latency > 30ms
        uint32_t nNeighborTxPackets{0};
        uint32_t nNeighborTxDataPackets{0};
        uint32_t nNeighborAckedPackets{0};
    };

    /**
     * @param linkId the link ID
     * @return the obs of the link after the last access restore.
     */
    AccessInfo GetLastAccessInfo(uint8_t linkId);

  protected:
    void DoInitialize() override;
    void DoDispose() override;

    /**
     * Wipe all expired packets that is considered stale.
     */
    virtual void WipeAllExpiredPackets();

  private:
    /**
     * @param sta the STA
     * @param interval the given interval
     * @returns the probability of the given STA having two transmissions less than the given
     * interval
     */
    double GetProbOfTxDataIntervalLessThan(Mac48Address sta, Time interval) const;

    /**
     * Remove all neighbor STAs that have not been active for a long time.
     */
    void RemoveInactiveNeighborStas();

    Ptr<WifiMac> m_mac; //!< associated MAC
    Time m_duration;    //!< maximum duration of the data stored
    std::map<uint64_t /* UID */, SelfPacketInfo> m_packetMap; //!< packets sent by this STA
    std::map<uint8_t /* link ID */, SelfLinkInfo>
        m_linkMap; //!< each link's channel utilization of this STA
    std::unordered_map<Mac48Address, NeighborStaInfo, WifiAddressHash>
        m_neighborStaMap;                       //!< channel utilization of neighbor STAs
    std::map<AcIndex, MacQueueInfo> m_queueMap; //!< queue information of each AC
};

} // namespace ns3

#endif /* SENSING_MAC_H */
