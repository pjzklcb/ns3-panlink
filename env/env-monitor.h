/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef ENV_MONITOR_H
#define ENV_MONITOR_H

#include "packet-info.h"

#include "ns3/singleton.h"
#include "ns3/vr-app-module.h"
#include "ns3/wifi-module.h"

namespace ns3
{

/**
 * \ingroup panlink
 */
class EnvMonitor : public Singleton<EnvMonitor>
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
    std::map<uint64_t /* UID */, std::list<PacketInfo>> m_packetMap;
    std::map<uint32_t /* SeqN*/, std::vector<uint64_t /* UID */>> m_burstMap;
};

} // namespace ns3

#endif /* ENV_MONITOR_H */
