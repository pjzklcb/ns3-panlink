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
#include "ns3/phy-entity.h"

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
        std::vector<std::pair<Time /* tx begin time */, Time /* tx end time */>> phyTxTime;
        Time ackedTime{Seconds(0)};
        bool isExpired{false};
        bool isDropped{false};
        double txPowerW;
    };

    /**
     * The channel utilization of this STA on the given link.
     */
    struct SelfLinkInfo
    {
        std::list<uint64_t> pendingPackets; //!< pending packets
        std::deque<std::pair<uint64_t /* uid */, Time /* tx end time */>> channelUtil;
    };

    /**
     * The information of the queues of each AC.
     */
    struct SelfMacQueueInfo
    {
        std::list<uint64_t> expiredPackets; // list of expired packets
        std::list<uint64_t> droppedPackets; // list of dropped packets
        std::deque<uint64_t> traffics;
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
     * Information of the neighbor packets.
     */
    struct NeighborPacketInfo
    {
        uint32_t packetSize;
        AcIndex ac;
        uint8_t linkId;
        bool isData{false};
        std::vector<std::pair<Time /* rx begin time */, Time /* rx end time */>> phyRxTime;
        Time ackedTime{Seconds(0)};
        double rssi;
    };

    /**
     * Information of the neighbor STAs.
     */
    struct NeighborStaInfo
    {
        NeighborStaRole role{UNKNOWN};
        uint8_t linkId;
        std::deque<std::pair<uint64_t /* uid */, Time /* rx end time */>> channelUtil;
    };

    /**
     * Trace the packet that has been received by wifi PHY.
     *
     * @param context the string contains link ID
     * @param packet the received packet
     * @param rxPowerW a map of the received power for each band
     */
    void NotifyPhyRxBegin(std::string context,
                          Ptr<const Packet> p,
                          RxPowerWattPerChannelBand rxPowersW);
    /**
     * Trace the packet that finished reception by wifi PHY.
     *
     * @param context the string contains link ID
     * @param packet the received packet
     */
    void NotifyPhyRxEnd(std::string context, Ptr<const Packet> p);

    /**
     * Handles received frames not matter the STA they are sent to.
     *
     * @param mpdu the first MPDU if the received frame is an A-MPDU
     * @param hdr the MAC header of the frame
     * @param rssi the RSSI of the frame
     * @param linkId the ID of the link
     */
    void NotifyFrameInfo(Ptr<const WifiMpdu> mpdu, const WifiMacHeader& hdr, double rssi, uint8_t linkId);

    /**
     * Record the STA whose last transmission was successful.
     *
     * @param sta the MAC address of the STA
     * @param linkId the ID of the link
     */
    void NotifyLastTxOkSta(Mac48Address sta, uint8_t linkId);

    /**
     * @param linkId the link ID
     * @returns the number of neighbor STAs sent packets
     */
    uint32_t GetNNeighborStas(uint8_t linkId) const;

    /**
     * @param sta the mac address
     * @returns the role of the neighbor STA
     */
    NeighborStaRole GetNeighborStaRole(Mac48Address sta) const;

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
     * The info collected from the mac since the last access resume.
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
    };

    /**
     * @param linkId the link ID
     * @return the info of the link after the last access resume.
     */
    AccessInfo GetLastAccessInfo(uint8_t linkId);

    /**
     * The info of data packet that utilizes channel
     */
    struct ChannelUtilInfo
    {
        Time duration;
        Time txEndTime;
        Time ackedTime{Seconds(0)};
        double rssi{0};
    };

    /**
     * @param period the period
     * @param linkId the link ID
     * @return the channel utilization of the mac on the given link in the past period
     */
    std::vector<ChannelUtilInfo> GetSelfLinkChannelUtil(Time period, uint8_t linkId) const;
    /**
     * @param period the period
     * @return the application traffic arrival time of this mac
     */
    std::vector<Time> GetSelfAppTraffic(Time period) const;

    /**
     * @param sta the MAC address of the neighbor STA
     * @param period the period
     * @return the obs from the neighbor STA in the past period
     */
    std::vector<ChannelUtilInfo> GetNeighborChannelUtil(Mac48Address sta, Time period) const;
    /**
     * @param linkId the link ID
     * @return the list of MAC addresses of the neighbor STAs
     */
    std::vector<Mac48Address> GetNeighborMacAddress(uint8_t linkId) const;

  protected:
    void DoInitialize() override;
    void DoDispose() override;

    /**
     * Wipe all packets that is considered stale.
     */
    virtual void WipeAllStaledPackets();

    /**
     * Wipe all neighbor STAs that have not been active for a long time.
     */
    virtual void WipeAllInactiveNeighborStas();

  private:
    Ptr<WifiMac> m_mac; //!< associated MAC
    Time m_duration;    //!< maximum duration of the data stored

    // self info
    std::map<uint64_t /* uid */, SelfPacketInfo> m_packetMap; //!< packets sent by this STA
    std::map<uint8_t /* link ID */, SelfLinkInfo>
        m_linkMap; //!< each link's channel utilization of this STA
    std::map<AcIndex, SelfMacQueueInfo> m_queueMap; //!< queue information of each AC

    // neighbor info
    std::map<uint64_t /* uid */, NeighborPacketInfo> m_neighborPacketMap; //!< neighbor packets
    std::unordered_map<Mac48Address, NeighborStaInfo, WifiAddressHash>
        m_neighborStaMap;                           //!< channel utilization of neighbor STAs
};

} // namespace ns3

#endif /* SENSING_MAC_H */
