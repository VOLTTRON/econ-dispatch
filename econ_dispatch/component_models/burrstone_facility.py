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
                            "boilers",
                            "generators",
                            "pmax",
                            "import_peak_on_startup",
                            "parasite",
                            "demand_charge",
                            "minimum_import"
                        ])

class Component(ComponentBase):
    def __init__(self,
                 boilers=[],
                 generators=[],
                 pmax=0,
                 import_peak_on_startup=0,
                 parasite=0,
                 demand_charge=0,
                 minimum_import=0,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        self.boilers = boilers
        self.parameters["boilers"] = self.boilers
        self.generators = generators
        self.parameters["generators"] = self.generators
        self.pmax = pmax
        self.parameters['pmax'] = self.pmax
        self.import_peak_on_startup = import_peak_on_startup
        self.parameters['import_peak_on_startup'] = self.import_peak_on_startup
        self.parasite = parasite
        self.parameters['parasite'] = self.parasite
        self.demand_charge = demand_charge
        self.parameters['demand_charge'] = self.demand_charge
        self.minimum_import = minimum_import
        self.parameters['minimum_import'] = self.minimum_import

    def get_output_metadata(self):
        return []

    def get_input_metadata(self):
        return []

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        # if now == self.reset_hour:
        #     self.import_peak_on_startup = 0
        # else:
        #     self.import_peak_on_startup = component_loads['import_peak_{}'.format(self.name)]
        # self.parameters['import_peak_on_startup'] = self.import_peak_on_startup
        return {}