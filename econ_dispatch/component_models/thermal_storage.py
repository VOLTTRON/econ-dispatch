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

from econ_dispatch.component_models import ComponentBase

LOG = logging.getLogger(__name__)

DEFAULT_MFR_CHWBLDG = 30.0  # kg/s
CP_FLUID_MAP = {"water": 4.186, "glycolwater30": 3.913, "glycolwater50": 3.558}

EXPECTED_PARAMETERS = set(["heat_cap", "max_power", "eff", "soc"])


class Component(ComponentBase):
    """Simple thermal storage model

    :param tank_volume:
    :param design_chilled_water_return_temp:
    :param design_chilled_water_supply_temp:
    :param max_power:
    :param fluid_type: one of 'water', 'glycolwater30' or 'glycolwater50'
    :param eff:
    :param ch_point_name: optimization decision variable name
        for charging setpoint
    :param disch_point_name: optimization decision variable name
        for discharging setpoint
    :param kwargs:
    """

    def __init__(
        self,
        tank_volume=None,
        design_chilled_water_return_temp=None,
        design_chilled_water_supply_temp=None,
        max_power=None,
        fluid_type="water",
        eff=0.98,
        ch_point_name="storage_ch_{}_0",
        disch_point_name="storage_disch_{}_0",
        **kwargs,
    ):
        super(Component, self).__init__(**kwargs)
        self.ch_point_name = ch_point_name.format(self.name)
        self.disch_point_name = disch_point_name.format(self.name)

        # unused
        self.chilled_water_supply_temp = None
        self.chilled_water_return_temp = None
        self.building_chilled_water_supply_temp = None
        self.building_chilled_water_return_temp = None

        # may cause problems if optimization starts before co-simulator
        # self.soc = None
        self.soc = 0.0

        if fluid_type not in CP_FLUID_MAP:
            LOG.warning("Unrecognized fluid type {}." " Defaulting to water.".format(fluid_type))
            fluid_type = "water"
        cp_fluid = CP_FLUID_MAP[fluid_type]

        unit_capacity_kWh = cp_fluid * abs(design_chilled_water_return_temp - design_chilled_water_supply_temp) / 3600.0

        heat_capacity = tank_volume * unit_capacity_kWh / 293.3  # to mmBTU

        if max_power is None:
            max_power = DEFAULT_MFR_CHWBLDG * unit_capacity_kWh

        self.heat_capacity = heat_capacity
        self.eff = eff
        self.max_power = max_power
        self.parameters["heat_cap"] = heat_capacity
        self.parameters["eff"] = eff
        self.parameters["max_power"] = max_power

    def validate_parameters(self):
        """Ensure that at least the necessary parameters are being passed
        to the optimizer (i.e., the model has been trained)"""
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def process_input(self, timestamp, name, value):
        """Process state of charge information from message bus.

        :param timestamp: time input data was published to the message bus
        :type timestamp: datetime.datetime
        :param name: name of the input from the configuration file
        :param value: value of the input from the message bus
        """
        if name == "soc":
            self.soc = value
            self.parameters["soc"] = value
        else:
            LOG.warning("heat capacity is not updated with true water temps")

    def get_mapped_commands(self, optimization_output):
        """Return the new set points on the device based on the received
        optimization output and the current state of the component.

        :param optimization_output: full output from optimizer solution
        :type optimization_output: dict
        :returns: map of name, command pairs to be mapped to device topics
        :rtype: dict
        """
        try:
            charge_load = optimization_output[self.ch_point_name]
            discharge_load = optimization_output[self.disch_point_name]
        except KeyError:
            LOG.warning("Thermal Storage missing from optimizer output")
            return {}
        return {"charge_load": charge_load * 293.3, "discharge_load": discharge_load * 293.3}
