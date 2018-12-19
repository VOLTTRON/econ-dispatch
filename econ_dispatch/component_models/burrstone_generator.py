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
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

EXPECTED_PARAMETERS = set([
                            "fundata",
                            "min_on",
                            "min_off",
                            "capacity",
                            "maint_cost",
                            "command_history"
                          ])

class Component(ComponentBase):
    def __init__(self,
                 min_on=0,
                 min_off=0,
                 capacity=0,
                 maint_cost=0,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        self.min_on = min_on
        self.parameters['min_on'] = self.min_on
        self.min_off = min_off
        self.parameters['min_off'] = self.min_off
        self.capacity = capacity
        self.parameters['capacity'] = self.capacity
        self.maint_cost = maint_cost
        self.parameters['maint_cost'] = self.maint_cost

        self.output = 0
        self.command_history = [0] * 24
        self.parameters['command_history'] = self.command_history

    def get_output_metadata(self):
        return [u"electricity", u"heated_water", u"steam"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        set_point = component_loads["generator_elec_{}_0".format(self.name)]

        self.output = set_point
        self.parameters["output"] = self.output

        # Do not update command history while testing
        # self.command_history = self.command_history[1:] + [int(set_point>0)]
        self.parameters["command_history"] = self.command_history[:]

        return {"command": int(set_point>0)}
