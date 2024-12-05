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

struct Obs
{
    // mechanism
    uint32_t nodeId;
    uint32_t agentType;
    uint32_t counter;
    double simTime;

    // self
    uint8_t ac;
    std::array<double, 4> thpt;    // actual throughput for different AC
    std::array<double, 4> thptReq; // {AC_BE, AC_BK, AC_VI, AC_VO}
    std::array<double, 4> delay;
    std::array<double, 4> delayReq;
    double selfChannelUtil;
    uint32_t nInQueuePackets;
    uint32_t nExpiredPackets;
    uint32_t nDroppedPackets;
    uint32_t nSendingPackets;
    uint32_t nAckedPackets;
    uint32_t nLowDelayPackets;
    uint32_t nHighDelayPackets;
    uint32_t cw;
    uint32_t nDeferredAccess;
    uint32_t waitedSlots;

    // neighbor
    uint32_t nBssStas;
    uint32_t nObssStas;
    std::array<uint32_t, 6> nTxOkDelay;
    std::array<double, 6> nExpectedContentionStas;
    double bssRssi;
    double obssRssi;
    double bssChannelUtil;
    double otherChannelUtil;
};

struct Act
{
    double ccaSensitivity;
    uint32_t backoff;
};

#endif // OBS_ACT_LIST_H
