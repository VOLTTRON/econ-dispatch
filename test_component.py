# -*- coding: utf-8 -*- {{{
# vim: set fenc=utf-8 ft=python sw=4 ts=4 sts=4 et:

# Copyright (c) 2018, Battelle Memorial Institute
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

logging.basicConfig(level=logging.DEBUG)

_log = logging.getLogger(__name__)

import argparse
import csv
import json
from ast import literal_eval
from econ_dispatch.component_models import get_component_class
from pprint import pprint

from dateutil.parser import parse

def main(component_name, csv_input_file, component_config_file=None):
    input_csv = csv.DictReader(csv_input_file)
    klass = get_component_class(component_name)

    kwargs = {}
    if component_config_file is not None:
        kwargs = json.loads(component_config_file.read())

    component = klass(**kwargs)

    for record in input_csv:
        parsed_record = {}
        if "timestamp" in record:
            parsed_record["timestamp"] = parse(record.pop("timestamp"))
        parsed_record.update((key,literal_eval(value)) for key,value in record.iteritems())
        component.update_parameters(**parsed_record)
        opt_params = component.get_optimization_parameters()
        print "Input:"
        pprint(parsed_record)
        print
        print "Output:"
        pprint(opt_params)
        print




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("component", help="Name of the component to test.")
    parser.add_argument("csv", type=argparse.FileType("rb"), help="CSV input file to load")
    parser.add_argument("--config-file", type=argparse.FileType("r"), help="Component configuration in JSON format")
    args = parser.parse_args()
    main(args.component, args.csv, args.config_file)
