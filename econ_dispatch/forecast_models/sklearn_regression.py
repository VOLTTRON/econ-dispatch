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
import json
import logging
import os
import pickle
from importlib import import_module

import pandas as pd
from volttron.platform.agent.utils import format_timestamp

from econ_dispatch.forecast_models.history import Forecast as HistoryForecastBase
from econ_dispatch.forecast_models.utils import make_time_features

LOG = logging.getLogger(__name__)


def my_import(name):
    """Import submodule by name

    :param name: full module name, e.g., sklearn.linear_models.Ridge
    :returns: submodule, e.g., Ridge
    """
    components = name.split(".")
    mod = import_module(".".join(components[:-1]))
    klass = getattr(mod, components[-1])
    return klass


class Forecast(HistoryForecastBase):
    """Return forecasts from scikit-learn regression on historical data

    :param dependent_variables: historical variables to regress on
    :param model_name: name of module with scikit-learn regression interface
    :param model_settings: keyword arguments for model
    :param kwargs: keyword arguments for base class
    """

    def __init__(
        self,
        dependent_variables=[],
        model_name="sklearn.linear_models.Ridge",
        model_settings={},
        serialize_on_train=False,
        output_dir=None,
        **kwargs,
    ):
        super(Forecast, self).__init__(**kwargs)
        if isinstance(dependent_variables, str):
            dependent_variables = [dependent_variables]
        self.dependent_variables = dependent_variables
        if model_name.split(".")[0] != "sklearn":
            raise NotImplementedError("Only sklearn models are supported")
        self.model = my_import(model_name)(**model_settings)

        self.independent_variables = []
        self.use_timestamp = False
        self.epoch = None
        self.epoch_span = None

        self.serialize_on_train = serialize_on_train
        self.output_dir = None
        if self.serialize_model and output_dir is None:
            raise ValueError("Specify an output directory to serialize the model on training")
        else:
            self.output_dir = os.path.expanduser(output_dir)

    def train(self, training_data):
        """Train regression model on historical data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        # load and preprocess
        super(Forecast, self).train(training_data)
        # remove NaNs
        self.historical_data = self.historical_data.loc[~self.historical_data.isnull().any(axis=1)]
        # project timestamps into vector space
        if self.timestamp_column in self.historical_data.columns:
            self.use_timestamp = True
            ts = self.historical_data.set_index(self.timestamp_column).index
            self.epoch = min(ts)
            self.epoch_span = float((max(ts) - self.epoch).total_seconds())
            time_features = make_time_features(
                ts, index=self.historical_data.index, epoch=self.epoch, epoch_span=self.epoch_span
            )
            self.historical_data = pd.concat([self.historical_data, time_features], axis=1)
            self.historical_data.drop(self.timestamp_column, axis=1, inplace=True)
        # leave all other variables independent
        self.independent_variables = [
            name for name in self.historical_data.columns if name not in self.dependent_variables
        ]

        self.model.fit(self.historical_data[self.independent_variables], self.historical_data[self.dependent_variables])

        # release historical data to save on memory
        # note that python garbage collection is not instantaneous
        LOG.warn("Releasing building load forecast training data. The agent will not be able to retrain on this data")
        self.historical_data = None

        if self.serialize_on_train:
            self.serialize_model(self.output_dir)

    def serialize_model(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config_file = os.path.join(output_dir, "config.json")
        if os.path.exists(config_file):
            LOG.warning("Overwriting serialized model config")
        model_file = os.path.join(output_dir, "sklearn_model.pkl")
        if os.path.exists(model_file):
            LOG.warning("Overwriting serialized model")

        LOG.debug(f"Serializing model to {output_dir}")
        with open(config_file, "w") as fh:
            json.dump(
                {
                    "use_timestamp": self.use_timestamp,
                    "epoch": format_timestamp(self.epoch),
                    "epoch_span": self.epoch_span,
                    "dependent_variables": self.dependent_variables,
                    "independent_variables": self.independent_variables,
                    "protocol": 5,
                    "sklearn_version": import_module("sklearn").__version__,
                },
                fh,
            )
        with open(model_file, "wb") as fh:
            pickle.dump(self.model, fh, protocol=5)

    def derive_variables(self, now, weather_forecast={}):
        """Predict forecast using regression model

        :param now: time of forecast
        :type now: datetime.datetime
        :param weather_forecast: dict containing a weather forecast
        :returns: dict of forecasts for time `now`
        """
        # project timestamps into vector space
        if self.use_timestamp:
            time_features = make_time_features(now, epoch=self.epoch, epoch_span=self.epoch_span)
            weather_forecast.update(time_features)
        X = pd.DataFrame(weather_forecast, index=[0])

        # Only ever see one record a time: pop values from 2D array
        y = self.model.predict(X[self.independent_variables])[0]
        result = {k: v for k, v in zip(self.dependent_variables, y)}
        return result
