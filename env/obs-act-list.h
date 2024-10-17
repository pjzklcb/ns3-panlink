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
    uint32_t txNode;
    uint32_t mcs;
    double holDelay;
    double throughput;
};

struct Act
{
    double newCcaSensitivity;
};

#endif // OBS_ACT_LIST_H
