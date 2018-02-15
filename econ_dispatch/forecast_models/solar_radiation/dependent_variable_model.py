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

import datetime
import numpy as np
import pandas as pd

import pandas as pd
from econ_dispatch.forecast_models import ForecastModelBase

# Magic numbers from one of the excel files
training_data = [25, 45, 65, 85, 95]


class Model(ForecastModelBase):
    def __init__(self, training_csv=None):
        self.training_csv = training_csv
        self.model1, self.model2 = self._training()

    def derive_variables(self, now, independent_variable_values={}):

        cloud_cover = independent_variable_values["Total Sky Cover {.1}"]
        cloud_cover_yesterday = independent_variable_values["Total Sky Cover {.1}-24"]
        irradiance_yesterday = independent_variable_values["Horizontal Infrared Radiation Intensity from Sky {Wh/m2}-24"]

        solar_radiation = self._deployment(now, cloud_cover, cloud_cover_yesterday, irradiance_yesterday)
        
        return {"solar_radiation": solar_radiation}

    def add_training_data(self, now, variable_values={}):
        """Do nothing for now."""
        pass

    def _training(self):
        #Reading radiation, cloud cover and time data from TMY3 excel file. Note
        #that TMY3 files do not contain Month variable
        data = pd.read_csv(self.training_csv, header=0)
        Ttmp = data["Time (HH:MM)"].values
        for i in range(len(Ttmp)):
            s = Ttmp[i]
            s = s[:s.index(':')]
            Ttmp[i] = float(s)

        Itmp = data["GHI (W/m^2)"].values
        CCtmp_x = data["TotCld (tenths)"].values
        means_xx = training_data

        #----------------------------------------------------------------------
        #Generating zero matrices. This way putting values in these matrices would
        #be faster in subsequent parts of the code.
        Wtmp = np.zeros(8760)
        CC1tmp = np.zeros(8760)
        CC2tmp = np.zeros(8760)
        CC24tmp = np.zeros(8760)
        I1tmp = np.zeros(8760)
        I2tmp = np.zeros(8760)
        I24tmp = np.zeros(8760)
        Id = np.zeros(8736) + 1
        a = [0, 720, 1392, 2136, 2856, 3600, 4320, 5064, 5808, 6528, 7272, 7992]
        b = [720, 1392, 2136, 2856, 3600, 4320, 5064, 5808, 6528, 7272, 7992, 8736]

        #-----------------------------------------------------------------
        #calculating the daytime/night time indicator
        for i in range(8760):
            if Ttmp[i] >= 8 and Ttmp[i] <= 19:
                Wtmp[i] = 1

        #---------------------------------------------------------------------
        #Transforming cloud cove values based on the means given
        CCtmp = np.zeros(8760)
        for i in range(8760):
            absvalues = np.absolute(means_xx - 10 * CCtmp_x[i])
            minimum = np.amin(absvalues) # Nick's Notes: I don't follow the operation going on here
            index = np.argmax(absvalues == minimum)
            CCtmp[i] = means_xx[index] # Nick's Notes: Don't follow this either

        #-------------------------------------------------------------------------
        #calculating the first lag of cloud cover and radiation
        CC1tmp[1:8760] = CCtmp[0:-1]
        I1tmp[1:8760] = Itmp[0:-1]

        #-------------------------------------------------------------------
        #calculating the second lag of cloud cover and radiation
        CC2tmp[2:8760] = CCtmp[0:-2]
        I2tmp[2:8760] = Itmp[0:-2]

        #-------------------------------------------------------------------
        #calculating the seasonal lag of cloud cover and radiation
        CC24tmp[24:8760] = CCtmp[0:8736]
        I24tmp[24:8760] = Itmp[0:8736]

        #------------------------------------------------------------------
        #Removing the first 24 rows as they do not contain full lag values
        I = Itmp[24:]
        CC = CCtmp[24:]
        I1 = I1tmp[24:]
        I2 = I2tmp[24:]
        I24 = I24tmp[24:]
        CC1 = CC1tmp[24:]
        CC2 = CC2tmp[24:]
        CC24 = CC24tmp[24:]
        W = Wtmp[24:]
        T = Ttmp[24:]

        #------------------------------------------------------------------------
        #forming training sets
        x1 = np.column_stack((Id, T, CC, CC24, I24, CC1, CC2, I1, I2))
        x2 = np.column_stack((Id, T, CC, CC24, I24))

        #fitting The models and finding the coefficients
        model1 = np.zeros((12, 9))
        model2 = np.zeros((12, 5))
        for i in range(12):
            W_diag = np.diag(W[a[i]:b[i]])
            W_diag = np.sqrt(W_diag)
            Bw = np.dot(I[a[i]:b[i]], W_diag)

            Aw = np.dot(W_diag, x1[a[i]:b[i]])
            X, residuals, rank, s = np.linalg.lstsq(Aw, Bw)
            model1[i] = X

            Aw = np.dot(W_diag, x2[a[i]:b[i]])
            X, residuals, rank, s = np.linalg.lstsq(Aw, Bw)
            model2[i] = X

        return model1, model2

    def _deployment(self, now, cloud_cover, cloud_cover_yesterday, irradiance_yesterday):

        hour_of_day = now.hour

        # the original model would only predict values for times between 8:00 and 19:00
        if hour_of_day < 8 or hour_of_day > 19:
            return 0

        # select model and adjust for indexes by zero
        mo2 = self.model2[now.month - 1]

        data = np.array([1,
                         hour_of_day,
                         cloud_cover,
                         cloud_cover_yesterday,
                         irradiance_yesterday])

        pr24 = max(np.dot(data, mo2), 0)

        return pr24
