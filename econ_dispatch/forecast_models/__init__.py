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
""".. todo:: Module docstring"""
import abc
from importlib import import_module
import logging
import pkgutil

LOG = logging.getLogger(__name__)


class ForecastBase(object, metaclass=abc.ABCMeta):
    """Abstract base class for forecast models

    :param training_window: period in days over which to train
    :param training_sources: dict of historian topic, name pairs
    """

    def __init__(self, training_window=365, training_sources={}):
        self.training_window = int(training_window)
        self.training_sources = training_sources

    @abc.abstractmethod
    def derive_variables(self, now, weather_forecast={}):
        """Return forecast for a single time, based on the weather forecast

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        pass

    def train(self, training_data):
        """Override this to use training data to update the model

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        pass


FORECAST_LIST = [x for _, x, _ in pkgutil.iter_modules(__path__)]
FORECAST_DICT = {}
for FORECAST_NAME in FORECAST_LIST:
    try:
        module = import_module(".".join(["econ_dispatch", "forecast_models", FORECAST_NAME]))
        klass = module.Forecast
    except Exception as e:
        LOG.error("Module {name} cannot be imported. Reason: {ex}" "".format(name=FORECAST_NAME, ex=e))
        continue

    # Validation of algorithm class
    if not issubclass(klass, ForecastBase):
        LOG.warning(
            "The implementation of {name} does not inherit from "
            "econ_dispatch.forecast_models.ForecastBase."
            "".format(name=FORECAST_NAME)
        )

    FORECAST_DICT[FORECAST_NAME] = klass


def get_forecast_class(name):
    """Return `Forecast` class from module named `name`"""
    return FORECAST_DICT.get(name)
