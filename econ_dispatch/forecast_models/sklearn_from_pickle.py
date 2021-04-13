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
import logging
import os
import pickle

import pandas as pd
from sklearn import __version__ as this_sklearn_version
from volttron.platform.agent.utils import process_timestamp

from econ_dispatch.forecast_models.history import Forecast as HistoryForecastBase
from econ_dispatch.forecast_models.utils import make_time_features

LOG = logging.getLogger(__name__)


class Forecast(HistoryForecastBase):
    """Return forecasts from pre-trained scikit-learn regression on historical data

    :param dependent_variables: historical variables to regress on
    :param model_settings: keyword arguments for model
    :param kwargs: keyword arguments for base class
    """

    def __init__(
        self,
        dependent_variables,
        independent_variables,
        use_timestamp,
        epoch,
        epoch_span,
        model_settings=None,
        **kwargs,
    ):
        super(Forecast, self).__init__(**kwargs)
        if model_settings is None:
            model_settings = {}
        self.dependent_variables = dependent_variables
        self.independent_variables = independent_variables
        self.use_timestamp = use_timestamp
        self.epoch, _ = process_timestamp(epoch)
        self.epoch_span = epoch_span

        self.model = self.load_serialized_model(**model_settings)

    def load_serialized_model(self, filepath, sklearn_version):
        """Load a pickled sklearn model from disk

        :param filepath: path to serialized sklearn model
        :param source_sklearn_version: version of sklearn used to train the model
        """
        if this_sklearn_version != sklearn_version:
            raise ValueError(
                f"This model was trained with sklearn version {sklearn_version}, but you're using "
                f"{this_sklearn_version}. Please install the correct version"
            )
        LOG.warning("Never unpickle data from an untrusted source. Beginning unpickle")
        with open(os.path.expanduser(filepath), "rb") as fh:
            model = pickle.load(fh)
        return model

    def train(self, training_data):
        """Re-train regression model on historical data

        :param training_data: data on which to train, organized by input name
        :type training_data: dict of lists
        """
        raise NotImplementedError("Pre-trained model cannot be re-trained")

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
