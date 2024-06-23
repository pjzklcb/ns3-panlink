/*
 * Copyright (c) 2024
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef PANLINK_ENV_MONITOR_H
#define PANLINK_ENV_MONITOR_H

#include "ns3/wifi-module.h"
#include "ns3/vr-app-module.h"


namespace ns3
{

/**
 * Record all the in-flight packet information
 */
struct InFlightPacketInfo
{
    Mac48Address m_srcAddress;
    Mac48Address m_dstAddress;
    AcIndex m_ac;
    uint32_t m_packetSize;
    Time m_edcaEnqueueTime{Seconds(0)}; // time the packet was enqueued into an EDCA queue
    Time m_edcaDequeueTime{Seconds(0)}; // time the packet was dequeued from EDCA queue
    Time appTypeTxTime{Seconds(0)};     // time the packet was created by app (E2E)
    Time appTypeRxTime{Seconds(0)};     // time the packet was received by app (E2E)
    Time m_HoLTime{Seconds(0)};         // time the packet became Head of Line
    Time m_L2RxTime{Seconds(0)};  // time packet got forwarded up to the Mac Layer (L2 latency =
                                  // L2RxTime - edcaEnqueueTime)
    Time m_phyTxTime{Seconds(0)}; // time packet began transmission
    bool m_dequeued{false};
};

/**
 * \ingroup panlink
 */
class PanlinkEnvMonitor : public Singleton<PanlinkEnvMonitor>
{
  public:
    void NotifyEdcaEnqueue(Ptr<const WifiMpdu> item);

    void NotifyAppTx(Ptr<const Packet> packet, const Address& address);

    void NotifyPhyTxBegin(std::string context, Ptr<const Packet> p, double txPowerW);

    void PhyTxDoneTrace(std::string context, Ptr<const Packet> p);

    void PhyTxDropTrace(std::string context, Ptr<const Packet> p);

    void NotifyMacForwardUp(Ptr<const Packet> p);

    void NotifyAppRx(Ptr<const Packet> packet, const Address& address);

    void NotifyMsduDequeuedFromEdcaQueue(Ptr<const WifiMpdu> item);

    void NotifyMsduDroppedAfterDequeueFromEdcaQueue(Ptr<const WifiMpdu> item);

    void FragmentTx(Ptr<const Packet> fragment,
                    const Address& from,
                    const Address& to,
                    const SeqTsSizeFragHeader& header);
    void FragmentRx(Ptr<const Packet> fragment,
                    const Address& from,
                    const Address& to,
                    const SeqTsSizeFragHeader& header);
    
    void BurstTx(Ptr<const Packet> burst,
                  const Address& from,
                  const Address& to,
                  const SeqTsSizeFragHeader& header);

    void BurstRx(Ptr<const Packet> burst,
                  const Address& from,
                  const Address& to,
                  const SeqTsSizeFragHeader& header);

    void CwTrace(std::string context, uint32_t cw, uint8_t /* linkId */);

    void BackoffTrace(std::string context, uint32_t newVal, uint8_t /* linkId */);

  protected:

  private:
    std::map<uint64_t /* UID */, std::list<InFlightPacketInfo>> m_packetMap;
    std::map<uint32_t /* SeqN*/, std::vector<uint64_t /* UID */>> m_burstMap;
};

} // namespace ns3

#endif /* PANLINK_ENV_MONITOR_H */
