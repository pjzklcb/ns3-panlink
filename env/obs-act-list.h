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

    uint32_t mcs;
    double holDelay;
    double throughput;
};

struct Act
{
    double ccaSensitivity;
};

#endif // OBS_ACT_LIST_H
