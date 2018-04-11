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
from econ_dispatch.forecast_models import HistoryModelBase
from econ_dispatch.system_model import build_model_from_config
from econ_dispatch.utils import normalize_training_data
#import networkx

import datetime as dt
import time
import re
from pprint import pformat
import csv

import numpy as np

from dateutil.parser import parse

_log = logging.getLogger(__name__)

#Config comment stripper taken from VOLTTRON.
_comment_re = re.compile(
    r'((["\'])(?:\\?.)*?\2)|(/\*.*?\*/)|((?:#|//).*?(?=\n|$))',
    re.MULTILINE | re.DOTALL)

def _repl(match):
    """Replace the matched group with an appropriate string."""
    # If the first group matched, a quoted string was matched and should
    # be returned unchanged.  Otherwise a comment was matched and the
    # empty string should be returned.
    return match.group(1) or ''


def strip_comments(string):
    """Return string with all comments stripped.
    Both JavaScript-style comments (//... and /*...*/) and hash (#...)
    comments are removed.
    """
    return _comment_re.sub(_repl, string)

def parse_json_config(config_str):
    """Parse a JSON-encoded configuration file."""
    return json.loads(strip_comments(config_str))

class InputHistory(HistoryModelBase):
    pass

class NullInputData(object):
    def __init__(self, history_data_file=None, historical_data_time_column="timestamp"):
        pass
    def derive_variables(self, now, independent_variable_values={}):
        return {}

def main(config_file, start, end,
         write_dot,
         input_csv_file_name,
         output_csv_file):
    overall_start_time = time.time()
    config = parse_json_config(config_file.read())
    model = build_model_from_config(config["weather"],
                                    config["optimizer"],
                                    config["components"],
                                    config["forecast_models"],
                                    config.get("optimization_frequency", 60),
                                    config.get("optimizer_debug"))

    # if write_dot:
    #     networkx.drawing.nx_pydot.write_dot(model.component_graph, config_file.name + ".dot")

    if input_csv_file_name is None:
        input_data = NullInputData()
    else:
        input_data = InputHistory()
        training_data = normalize_training_data(input_csv_file_name)
        input_data.train(training_data)

    now = start
    time_step = dt.timedelta(hours=1)

    run_times = []

    results = []

    try:
        while now < end:
            _log.debug("Processing timestamp: " + str(now))
            start_time = time.time()
            result = model.run(now, input_data.derive_variables(now))
            if result:
                results.append((now, result))
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

        if output_csv_file is not None:
            if results:
                topics = set()
                for result in results:
                    for topic in result[1]:
                        topics.add(topic)

                topics = list(topics)
                topics.sort()
                topics = ["timestamp"] + topics
                _log.info("Unique commands:\n"+pformat(topics))

                dict_writer = csv.DictWriter(output_csv_file, topics)
                dict_writer.writeheader()
                for result in results:
                    row = {}
                    row["timestamp"] = result[0]
                    for topic, value in result[1].iteritems():
                        row[topic] = value

                    dict_writer.writerow(row)
            else:
                _log.info("Results empty, no output file created.")
        else:
            _log.info("No output file specified.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=argparse.FileType("r"), help="Configuration file to load")
    parser.add_argument("--start-time", default="2017-1-1", help="Start date time in ISO format")
    parser.add_argument("--end-time", default="2017-12-29", help="End date time in ISO format")
    parser.add_argument("--input-data", help="Sensor input data as a CSV file")
    parser.add_argument("--output-data", type=argparse.FileType("wb"), help="Device commands generated by the model")
    parser.add_argument("--write-dot", default=False, action="store_const", const=True,
                        help="Write graphviz dot file for configuration.")

    args = parser.parse_args()

    start = parse(args.start_time)
    end = parse(args.end_time)

    _log.info("Simulation start: "+str(start))
    _log.info("Simulation end: "+str(end))

    main(args.config, start, end,
         args.write_dot,
         args.input_data,
         args.output_data)