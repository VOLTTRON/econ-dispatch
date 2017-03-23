# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2017, Battelle Memorial Institute
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

from econ_dispatch.component_models import ComponentBase
from econ_dispatch.building_load_models import get_model_class
from econ_dispatch.weather.weather import WeatherPrediction
import datetime as dt

class Component(ComponentBase):
    def __init__(self, building_model_type="", building_model_settings={}, weather_model_settings={}, **kwargs):
        super(Component, self).__init__(**kwargs)
        model_class = get_model_class(building_model_type)
        self.building_model = model_class(dependent_variables=["heat_load", "cool_load", "elec_load"],
                                          **building_model_settings)
        self.weather = WeatherPrediction(**weather_model_settings)

        self.heat_loads = [0.0] * 24
        self.cool_loads = [0.0] * 24
        self.elec_loads = [0.0] * 24

    def get_input_metadata(self):
        return [u"heated_air", u"cooled_air", u"electricity"]

    def get_optimization_parameters(self):
        return {"heat_load":self.heat_loads, "cool_load":self.cool_loads, "elec_load":self.elec_loads}

    def update_parameters(self, timestamp=None, **kwargs):
        #Skip this if we are setting our initial state.
        if timestamp is None:
            return
        #Update the state of the prediction.
        timestamp += dt.timedelta(hours = 1)
        weather_records = self.weather.get_weather_data(timestamp)

        self.heat_loads = hl = []
        self.cool_loads = cl = []
        self.elec_loads = el = []

        for wr in weather_records:
            bl = self.building_model.derive_variables(timestamp, wr)
            hl.append(bl["heat_load"])
            cl.append(bl["cool_load"])
            el.append(bl["elec_load"])
            timestamp += dt.timedelta(hours=1)



