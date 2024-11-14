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
    std::array<double, 4> thpt;    // actual throughput for different AC
    std::array<double, 4> thptReq; // {AC_BE, AC_BK, AC_VI, AC_VO}
    std::array<double, 4> delay;
    std::array<double, 4> delayReq;
    double selfD2lt;
    double selfChannelUtil;
    double ackRatio; // ratio of acked MPDUs
    uint8_t ac;
    uint32_t cw;
    uint32_t nDeferredAccess;
    uint32_t waitedSlots;

    // neighbor
    uint32_t nBssStas;
    uint32_t nObssStas;
    double bssRssi;
    double obssRssi;
    double bssD2lt;
    double otherD2lt;
    double bssChannelUtil;
    double otherChannelUtil;
};

struct Act
{
    double ccaSensitivity;
    double backoff;
};

#endif // OBS_ACT_LIST_H
