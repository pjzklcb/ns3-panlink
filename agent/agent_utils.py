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
        self.role = obs.role
        self.period = obs.period
        self.n_acks = obs.nAcks
        self.txok_delay = obs.lastTxOkDelay
        self.txok_idle_time = obs.lastTxOkIdleTime
        self.rssi = obs.meanRssi
        self.txok_interval_stats = [v for v in obs.txOkIntervalStats]
        self.backoff = obs.backoffSlots
        self.cw = [v for v in obs.cw]


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.std = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        x = np.array(x)
        batch_mean = np.mean(x, axis=0)
        batch_std = np.std(x, axis=0)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_std, batch_count)

    def update_from_moments(self, batch_mean, batch_std, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        new_var = (self.count * (self.std**2) + batch_count * (batch_std**2) +
                   (delta**2) * self.count * batch_count / tot_count) / tot_count
        new_std = np.sqrt(new_var)

        self.mean = new_mean
        self.std = new_std
        self.count = tot_count


class Normalizer(object):
    def __init__(self, shape=()):
        self.running_mean_std = RunningMeanStd(shape=shape)

    def normalize(self, x):
        self.running_mean_std.update(x)
        return (x - self.running_mean_std.mean) / self.running_mean_std.std

    def denormalize(self, x):
        return x * self.running_mean_std.std + self.running_mean_std.mean


class ExponentialScaler:
    def __init__(self, alpha=0.1):
        self.min_val = None
        self.max_val = None
        self.alpha = alpha  # EMA update weight

    def update(self, x):
        # x shape: (batch_size, input_dim)
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        if self.min_val is None or self.max_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            self.min_val = self.alpha * x_min + (1 - self.alpha) * self.min_val
            self.max_val = self.alpha * x_max + (1 - self.alpha) * self.max_val

    def normalize(self, x):
        self.update(x)
        return 2 * (x - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1

    def denormalize(self, x):
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val


class MinMaxScaler:
    def __init__(self, neighbor=False):
        if neighbor:
            self.min_val = np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.30000000e-04,
                                     -3.35230000e-02, 0.00000000e+00, -6.17058411e+01, 0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                                     0.00000000e+00, 0.00000000e+00, 0.00000000e+00])
            self.max_val = np.array([3.00000000e+00, 7.30000000e+01, 7.30000000e+01, 5.75931460e+03,
                                     5.75931460e+03, 1.27790400e+02, 0.00000000e+00, 4.98642400e+02,
                                     3.08784905e+02, 4.98642400e+02, 4.98642400e+02, 3.23040000e+00,
                                     2.22144666e+00, 3.23040000e+00, 3.23040000e+00])
        else:
            self.min_val = np.array([0.   , 0.   , 0.044, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
                                     0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ])
            self.max_val = np.array([3.80000000e+01, 2.30000000e+01, 4.92628260e+03, 4.92628260e+03,
                                     2.49280000e+00, 1.35833901e+02, 1.35225475e+02, 2.21344480e+01,
                                     2.85367255e+02, 2.47420300e+02, 3.28343761e+02, 3.55546000e+01,
                                     4.79107600e+02, 6.56000000e-02, 4.24918736e-17, 6.56000000e-02,
                                     6.56000000e-02])

    def normalize(self, x):
        # x shape: (batch_size, input_dim)
        return 2 * (x - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1

    def denormalize(self, x):
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val