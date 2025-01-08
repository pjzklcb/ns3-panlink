/*
 * Copyright (c) 2024
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Authors: Pan Jinzhe <pjzklcb@gmail.com>
 */

#ifndef OBS_ACT_LIST_H
#define OBS_ACT_LIST_H

#include <array>
#include <cstdint>

/**
 * The obs collected from the mac on the given link from last interaction.
 */
struct LastAccessObs
{
    uint32_t nInQueuePackets;
    uint32_t nExpiredPackets;
    uint32_t nDroppedPackets;
    uint32_t nSendingPackets;
    uint32_t nAckedPackets;
    uint32_t nLowDelayPackets;
    uint32_t nHighDelayPackets;
};

/**
 * The obs collected from the mac on the given link in the last period.
 */
struct SelfLinkObs
{
    double period; // period of observation in ms

    uint32_t nTxDataPackets;
    uint32_t nAckedDataPackets;
    double lastTxOkDelay; // in ms
    double lastChannelFailDelay;
    double lastChannelFailIdleTime;
    uint32_t nChannelFail;
    double idleRatio;

    std::array<double, 4> trafficIntervalStats; // mean, std, min, max
    std::array<double, 4> txDataIntervalStats;
};

/**
 * The obs collected from neighbor STA in the last period.
 */
struct NeighborStaObs
{
    uint32_t nodeId;
    uint8_t role;
    double period; // period of observation in ms

    uint32_t nAcks;
    double lastTxOkDelay;    // in ms
    double lastTxOkIdleTime; // the total channel idle time from last ack
    double meanRssi;         // in dBm

    std::array<double, 4> txOkIntervalStats;

    uint32_t backoffSlots;
    std::array<uint32_t, 4> cw;
};

struct Obs
{
    // mechanism
    uint32_t nodeId;
    uint32_t agentType;
    uint32_t counter;
    bool moreData;
    double simTime;

    // self
    uint8_t ac;
    std::array<double, 4> thpt;    // actual throughput for different AC
    std::array<double, 4> thptReq; // {AC_BE, AC_BK, AC_VI, AC_VO}
    std::array<double, 4> delay;
    std::array<double, 4> delayReq;

    // self link
    SelfLinkObs selfLinkObs;
    LastAccessObs lastAccessObs;
    uint32_t cw;
    uint32_t nDeferredAccess;
    uint32_t waitedSlots;

    // neighbor
    uint32_t nNeighborStas;
    NeighborStaObs neighborStaObs;
    uint32_t nStasWithPackets;
    uint32_t minNeighborBackoff;
};

struct Act
{
    double ccaSensitivity;
    uint32_t backoff;
};

#endif // OBS_ACT_LIST_H
