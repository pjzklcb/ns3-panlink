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
