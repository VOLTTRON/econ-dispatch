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

import logging

logging.basicConfig(level=logging.DEBUG)

import argparse
import json
from econ_dispatch.application import Application
import networkx

import datetime as dt
import time

import numpy as np

from dateutil.parser import parse

_log = logging.getLogger(__name__)

def main(config_file, start, end, write_dot):
    overall_start_time = time.time()
    config = json.loads(config_file.read())
    application = Application(model_config=config)

    print application.model

    if write_dot:
        networkx.drawing.nx_pydot.write_dot(application.model.component_graph, config_file.name + ".dot")

    now = start
    time_step = dt.timedelta(hours=1)

    run_times = []


    try:
        while now < end:
            _log.debug("Processing timestamp: " + str(now))
            start_time = time.time()
            application.run(now, {})
            now += time_step
            end_time = time.time()
            run_times.append(end_time-start_time)
    finally:
        if run_times:
            run_time_array = np.array(run_times)
            mean = run_time_array.mean()
            std = run_time_array.std()
            max_time = run_time_array.max()
            total_time = time.time() - overall_start_time
            _log.info("Total Run Time: " + str(total_time))
            _log.info("Application Run Average: " + str(mean))
            _log.info("Application Run Standard Deviation: " + str(std))
            _log.info("Application Run Max: " + str(max_time))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=argparse.FileType("r"), help="Configuration file to load")
    parser.add_argument("--start-time", default="2017-1-1", help="Start date time in ISO format")
    parser.add_argument("--end-time", default="2017-12-29", help="End date time in ISO format")
    parser.add_argument("--write-dot", default=False, action="store_const", const=True,
                        help="Write graphviz dot file for configuration.")

    args = parser.parse_args()

    start = parse(args.start_time)
    end = parse(args.end_time)

    _log.info("Simulation start: "+str(start))
    _log.info("Simulation end: "+str(end))

    main(args.config, start, end, args.write_dot)