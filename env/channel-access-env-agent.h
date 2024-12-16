/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef CHANNEL_ACCESS_ENV_AGENT_H
#define CHANNEL_ACCESS_ENV_AGENT_H

#include "env-agent.h"

namespace ns3
{

/**
 * \brief env agent for channel access
 * \ingroup panlink
 */

class ChannelAccessEnvAgent : public EnvAgent
{
  public:
    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    ChannelAccessEnvAgent();
    ~ChannelAccessEnvAgent() override;

    struct ChannelAccessObs : public EnvAgent::ModuleObs
    {
        AcIndex ac;
        uint32_t cw;
        uint32_t nDeferredAccess;
        uint32_t waitedSlots;
    };

    void GetEnvObs(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs) override;
    void SetEnvAct(Ns3AiMsgInterfaceImpl<Obs, Act>* msgInterface, ModuleObs& obs) override;

  private:
    typedef std::vector<SensingMac::ChannelUtilInfo> ChannelUtilList;

    /**
     * @param util a list of channel utilization information
     * @returns the number of ACKed data packets
     */
    uint32_t GetNAckedDataPackets(const ChannelUtilList& util) const;

    /**
     * @returns the delay from now to the last tx data packet
     */
    double GetTxDataDelay(const ChannelUtilList& util) const;

    /**
     * @returns the delay from now to the last acked data packet
     */
    double GetTxOkDelay(const ChannelUtilList& util) const;

    /**
     * @returns the mean, std, min, max of the data arrival intervals
     */
    std::array<double, 4> GetDataArrivalIntervalStats(const ChannelUtilList& util) const;

    /**
     * @returns the mean, std, min, max of the tx data intervals
     */
    std::array<double, 4> GetTxDataIntervalStats(const ChannelUtilList& util) const;

    /**
     * @returns the channel utilization time in ms
     */
    double GetChannelUtil(const ChannelUtilList& util) const;

    /**
     * @returns the mean rssi of the tx packets.
     */
    double GetMeanRssi(const ChannelUtilList& util) const;

    /**
     * @param vec a vector of double
     * @returns the mean, std, min, max
     */
    std::array<double, 4> GetStats(const std::vector<double>& vec) const;

    uint32_t m_idxNeighborSta; //!< current index of neighbor sta
};

} // namespace ns3

#endif /* CHANNEL_ACCESS_ENV_AGENT_H */
