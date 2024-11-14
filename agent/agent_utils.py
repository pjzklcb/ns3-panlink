# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

from enum import Enum

class AgentType(Enum):
    NO_RL = 0
    RL_CCA_SENSITIVITY = 1
    RL_CHANNEL_ACCESS = 2
    RL_RATE_CONTROL = 3
    RL_AGGREGATION = 4


class AcIndex(Enum):
    AC_BE = 0
    AC_BK = 1
    AC_VI = 2
    AC_VO = 3