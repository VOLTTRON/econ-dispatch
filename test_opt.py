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

import numpy as np
import pandas as pd
from econ_dispatch.optimizer.use_case_1 import optimize
from econ_dispatch.utils import natural_keys, OptimizerCSVOutput
# data = xlsread('Hospital Modeled Data.xlsx')
data = pd.read_csv("./hospital_modeled_data.csv", parse_dates=["Date/Time"])

# 6/26 both cooling and heating
range_low = 4226
range_high = 4250

E_PV = np.zeros(24)     #kW
E_load = data['Building Electric Load'].values[range_low:range_high] #kW

Q_loadheat = data['Building Heating Load'].values[range_low:range_high]
Q_loadheat /= 293.1  # kW -> mmBtu/hr

Q_loadcool = data['Building Cooling Load'].values[range_low:range_high]
Q_loadcool /= 293.1  # kW -> mmBtu/hr

# get the price info
lambda_gas = 7.614 * np.ones(24)   #$/mmBtu
lambda_elec_fromgrid = 0.1 * np.ones(24)  #$/kWh
lambda_elec_togrid = 0.1 * np.ones(24)  #$/kWh

forecast = []

for h in range(24):
    record = {"elec_load": E_load[h],
              "heat_load": Q_loadheat[h],
              "cool_load": Q_loadcool[h],
              "solar_kW": 0,
              "natural_gas_cost": 7.614,
              "electricity_cost": 0.1,
              }

    forecast.append(record)

results = optimize(forecast)

for result_key in sorted(results.keys(), key=natural_keys):
    print "{} = = {}".format(result_key, results[result_key])

csv_out = OptimizerCSVOutput("hospital_modeled_data_output.csv")

csv_out.writerow(results, forecast)

csv_out.close()