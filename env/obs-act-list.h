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
    std::array<double, 4> thpt;     // actual throughput for different AC
    std::array<double, 4> thptReq;  // {AC_BE, AC_BK, AC_VI, AC_VO}
    std::array<double, 4> delay;
    std::array<double, 4> delayReq;

    // neighbor
    double apRssi;
    double apTxPower;
    double bssRssi;
    double bssTxPower;
    double obssRssi;
    double obssTxPower;
};

struct Act
{
    double ccaSensitivity;
};

#endif // OBS_ACT_LIST_H
