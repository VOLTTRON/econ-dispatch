# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2019, Battelle Memorial Institute
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation
# are those of the authors and should not be interpreted as representing
# official policies, either expressed or implied, of the FreeBSD
# Project.
#
# This material was prepared as an account of work sponsored by an
# agency of the United States Government.  Neither the United States
# Government nor the United States Department of Energy, nor Battelle,
# nor any of their employees, nor any jurisdiction or organization that
# has cooperated in the development of these materials, makes any
# warranty, express or implied, or assumes any legal liability or
# responsibility for the accuracy, completeness, or usefulness or any
# information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or
# service by trade name, trademark, manufacturer, or otherwise does not
# necessarily constitute or imply its endorsement, recommendation, or
# favoring by the United States Government or any agency thereof, or
# Battelle Memorial Institute. The views and opinions of authors
# expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY
# operated by BATTELLE for the UNITED STATES DEPARTMENT OF ENERGY
# under Contract DE-AC05-76RL01830
# }}}
"""Utilities for forecast models"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.holiday import Holiday  # NOQA: F401


def make_time_features(ts, index=None, epoch=None, epoch_span=None):
    """Project datetimes into vector space for use in machine learning models

    Outputs:

    - projection onto the unit circle of
      - second of day
      - day of week
      - day of year
    - seconds since `epoch`, normalized by `epoch_span`
    - binary workday indicator (i.e., Monday-Friday except major US holidays)

    :param ts: timestamp(s) to process
    :type ts: datetime.datetime or iterable thereof
    :param index: index of ts (e.g., if from a larger dataframe)
    :param epoch: start of time reckoning
    :param epoch_span: length of time reckoning
    :rtype: pd.DataFrame
    :returns: various projections of datetimes into vector space
    """
    # input validation
    try:
        if len(ts) == 1:
            _singleton = True
        elif len(ts) > 1:
            _singleton = False
        elif len(ts) < 1:
            raise ValueError("must pass non-empty iterable of timestamps")
    except TypeError:
        return make_time_features([ts], index=index, epoch=epoch, epoch_span=epoch_span)

    if not isinstance(ts, pd.DatetimeIndex):
        ts = pd.Series(0, index=ts).index
    if not isinstance(ts, pd.DatetimeIndex):
        raise ValueError("must pass non-empty iterable of timestamps")

    if index is None:
        index = pd.RangeIndex(len(ts))
    if epoch is None:
        epoch = min(ts)
    if epoch_span is None:
        epoch_span = float((max(ts) - epoch).total_seconds())

    time_features = {}
    start = min(ts)
    end = max(ts)

    # Major US holidays
    NewYearsDay = pd.tseries.holiday.Holiday("New Years Day", month=1, day=1)
    MemorialDay = pd.tseries.holiday.Holiday("Memorial Day", month=6, day=1, offset=pd.DateOffset(weekday=MO(-1)))
    IndependenceDay = pd.tseries.holiday.Holiday("Independence Day", month=7, day=4)
    LaborDay = pd.tseries.holiday.Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1)))
    ThanksgivingDay = pd.tseries.holiday.Holiday(
        "Thanksgiving Day", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))
    )
    ChristmasDay = pd.tseries.holiday.Holiday("Christmas Day", month=12, day=25)
    holidays = (
        NewYearsDay.dates(start.date(), end.date()).tolist()
        + MemorialDay.dates(start.date(), end.date()).tolist()
        + IndependenceDay.dates(start.date(), end.date()).tolist()
        + LaborDay.dates(start.date(), end.date()).tolist()
        + ThanksgivingDay.dates(start.date(), end.date()).tolist()
        + ChristmasDay.dates(start.date(), end.date()).tolist()
    )
    holidays = set([h.date() for h in holidays])

    # projections onto unit circle
    time_features["day_cos"] = np.cos((ts.hour * 3600 + ts.minute * 60 + ts.second) * 2 * np.pi / 86400.0)
    time_features["day_sin"] = np.sin((ts.hour * 3600 + ts.minute * 60 + ts.second) * 2 * np.pi / 86400.0)
    time_features["week_cos"] = np.cos(ts.dayofweek * 2 * np.pi / 7.0)
    time_features["week_sin"] = np.sin(ts.dayofweek * 2 * np.pi / 7.0)
    time_features["year_cos"] = np.cos(ts.dayofyear * 2 * np.pi / 365.0)
    time_features["year_sin"] = np.sin(ts.dayofyear * 2 * np.pi / 365.0)
    # linear march through time
    time_features["epoch"] = (ts - epoch).total_seconds() / epoch_span
    # workday indicator
    time_features["workday"] = [int(weekday < 5 and date not in holidays) for weekday, date in zip(ts.weekday, ts.date)]

    if _singleton:
        return {k: v[0] for k, v in time_features.items()}
    else:
        return pd.DataFrame(time_features, index=index)
