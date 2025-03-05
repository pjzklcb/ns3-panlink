# Copyright (c) 2024
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Author: Pan Jinzhe <pjzklcb@gmail.com>

from enum import Enum
import numpy as np

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


class AccessInfo:
    def __init__(self, ac, cw, n_deferred_access, n_waited_slot, obs):
        self.ac = ac
        self.cw = cw
        self.n_deferred_access = n_deferred_access
        self.n_waited_slot = n_waited_slot
        self.n_in_queue = obs.nInQueuePackets
        self.n_expired = obs.nExpiredPackets
        self.n_dropped = obs.nDroppedPackets
        self.n_sending = obs.nSendingPackets
        self.n_acked = obs.nAckedPackets
        self.n_low_delay = obs.nLowDelayPackets
        self.n_high_delay = obs.nHighDelayPackets


class SelfLinkInfo:
    def __init__(self, obs):
        self.period = obs.period
        self.n_tx_data = obs.nTxDataPackets
        self.n_acked_data = obs.nAckedDataPackets
        self.txok_delay = obs.lastTxOkDelay
        self.chan_fail_delay = obs.lastChannelFailDelay
        self.chan_fail_idle_time = obs.lastChannelFailIdleTime
        self.n_chan_fail = obs.nChannelFail
        self.idle_ratio = obs.idleRatio
        self.app_interval_stats = [v for v in obs.trafficIntervalStats]
        self.tx_interval_stats = [v for v in obs.txDataIntervalStats]


class NeighborLinkInfo:
    def __init__(self, obs):
        self.node_id = obs.nodeId
        self.role = obs.role
        self.period = obs.period
        self.n_acks = obs.nAcks
        self.txok_delay = obs.lastTxOkDelay
        self.txok_idle_time = obs.lastTxOkIdleTime
        self.rssi = obs.meanRssi
        self.txok_interval_stats = [v for v in obs.txOkIntervalStats]
        self.backoff = obs.backoffSlots
        self.cw = [v for v in obs.cw]
