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

import json
import os
import numpy as np

from math import pi, sin, cos, tan, acos, asin, floor

from econ_dispatch.forecast_models import ForecastModelBase
from econ_dispatch.forecast_models import get_forecast_model_class



class Model(ForecastModelBase):
    def __init__(self,
                 dynamic_data_file=None,
                 history_data_file=None,
                 plane_tilt_angle=None,
                 plane_azimuth_angle=None,
                 local_latitude=None,
                 local_longitude=None,
                 time_zone_longitude=None,
                 solar_radiation_model_settings=None):

        self.dynamic_data_file = dynamic_data_file
        self.history_data_file = history_data_file

        self.plane_tilt_angle = plane_tilt_angle
        self.plane_azimuth_angle = plane_azimuth_angle
        self.local_latitude = local_latitude
        self.local_longitude = local_longitude
        self.time_zone_longitude = time_zone_longitude

        self.solar_radiation_model = get_forecast_model_class("solar_radiation",
                                                              "dependent_variable_model",
                                                              **solar_radiation_model_settings)

        # G&R model coefficients (Annual Model)
        self.a1, self.a2, self.a3 = self.train()

    def derive_variables(self, now, independent_variable_values={}):
        solar_kW = 0

        solar_radiation = self.solar_radiation_model.derive_variables(now, independent_variable_values)
        solar_radiation = solar_radiation["solar_radiation"]
        temperature = independent_variable_values["tempm"]

        solar_kW = self.predict(solar_radiation, temperature, now)

        return {"solar_kW": solar_kW}

    def add_training_data(self, now, variable_values={}):
        """Do nothing for now."""
        pass

    def predict(self, ITf, Taf, now):
        # *********** scenario 5- DC Power, ambient temperature, and IT are measured at-site ********
        # here we assumed that regression models were built separately and
        # therefore regression coefficients are available. also, forecasted values
        # for IT and Ta are assumed to be available. This code is meant to be used
        # for 24 hours ahead predictions- The code creates an excel file named
        # "power-Output" and write the results (DC Power) on it along with date and
        # time stamps

        # ******** Reading Forecasted Solar and Ambient Temperatures ************
        # data_file = os.path.join(os.path.dirname(__file__), 'solar_pv_dynamic_inputs.json')
        # with open(self.dynamic_data_file, 'r') as f:
        #     dynamic_inputs = json.load(f)

        # ITf = dynamic_inputs["ITf"] # Forecasted Total solar irradiance on tilted plane or POA irradiance, [W/m2]
        # Taf = dynamic_inputs["Taf"] # Forecasted ambient temperature, [Degrees C]
        # TOD = dynamic_inputs["time_placeholder"] # Time of the day as a numbers between 0 and 1
        # raw = dynamic_inputs["date"]# Dates as an array of strings

        # ******** PV Mount, User inputs *******
        theta_p = self.plane_tilt_angle #Plane tilt angle
        theta_p = theta_p * (pi / 180) #converting degrees to radians
        phi_p = self.plane_azimuth_angle #Plane azimuth angle
        phi_p = phi_p * (pi / 180)#converting degrees to radians



        # ******** Location Info, User Input OR obtain somehow from zip code or other smart means ************
        _lambda = self.local_latitude #Location latitude
        _lambda = _lambda * (pi / 180)#converting degrees to radians

        Lloc = self.local_longitude #Local longitude
        Lstd = self.time_zone_longitude #Time zone longitude


        # ********************************************************
        Pdc = 0 # PV power output, DC
        n = 0 # Day number
        theta_s = 0 # Zenith angle of sun
        cos_theta_s = 0 # cos of zenith angle
        cos_theta_i = 0 # cos of incidence angle
        phi_s = 0 # Azimuth of sun
        sin_phi_s = 0 # sin of sun azimuth angle
        delta = 0 # Solar declination
        t_std = 0 #Standard time
        t_sol = 0 # Solar Time
        omega = 0 # Hour angle
        Ketta = 0 # Incidence angle modifier
        omegaS = 0 # sunrise and sunset hour angle
        cos_omega_S = 0 # cos of omegaS
        daylightindicator = 0 # indicates whether sun is above the horizon or not
        omegaSprime = 0 # sunset and sunrise hour angle over the panel surface
        cos_omegaS_prime = 0 # cos of omegaSprime
        daylightindicatorT = 0 # indicates whether sun is above the edge of the panel or not (sunrise and sunset over the panel surface)

        #********* Calculating n (day number) **********
        # for a in range(24): # MIKE INDEXES
        date_string = now #converting date to string
        month = date_string.month
        day = date_string.day

        if month == 1:
            n = day
        elif month == 2:
            n = 31 + day
        elif month == 3:
            n = 59 + day
        elif month == 4:
            n = 90 + day
        elif month == 5:
            n = 120 + day
        elif month == 6:
            n = 151 + day
        elif month == 7:
            n = 181 + day
        elif month == 8:
            n = 212 + day
        elif month == 9:
            n = 243 + day
        elif month == 10:
            n = 273 + day
        elif month == 11:
            n = 304 + day
        elif month == 12:
            n = 334 + day

        TOD = now.hour / 24.0
        t_std = (TOD * 24) - 0.5 # hour ending data collection
        delta = -sin((pi/180)*23.45) * cos((pi/180)*360*(n+10)/365.25)

        # Solar Time
        B = 360 * (n - 81) / 364
        B = B * (pi / 180)
        Et = 9.87 * sin(2*B) - 7.53 * cos(B) - 1.5 * sin(B) # equation of time
        t_sol = t_std - ((Lstd - Lloc) / 15) + (Et / 60) # solar time in hr

        # Hour Angle
        omega = (t_sol - 12) * 15
        omega = omega * (pi / 180)
        cos_theta_s = cos(_lambda) * cos(delta) * cos(omega) + sin(_lambda) * sin(delta) # thetas is zenith angle of sun
        sin_theta_s = sin(pi/2-acos(cos_theta_s))
        cos_theta_s = abs(cos_theta_s)

        # Sunrise and sunset solar angle for horizontal surfaces
        if sin_theta_s < 0:
            DI = 0
        else:
            DI = 1

        # **************************************
        theta_s = acos(cos_theta_s) #thetas will be in radian
        sin_phi_s = (cos(delta) * sin(omega)) / sin(theta_s) #phis is azimuth of sun
        phi_s = asin(sin_phi_s)
        cos_theta_i = (sin(theta_s) * sin(theta_p) * cos(phi_s - phi_p)) + (cos(theta_s) * cos(theta_p)) # thetai is solar incidence angle on plane
        Ketta = 1 - 0.1 * ((1 / cos_theta_i) - 1) # incidence angle modifier

        # Power Calculation- G&R model
        Pdc = DI * (self.a1 * ITf * Ketta + self.a2 * ITf * Ketta * Taf + self.a3 * (ITf * Ketta)**2)

        return Pdc

    def train(self):
        # This module reads the historical data on ambient temperatures (in C), PV power
        # generation (in W) and POA irradiation (in W/m2); then, calculates the
        # ketta using time stamps of the historical data and fit the power prediction model.
        # At the end, regression coefficients will be written to a file.

        # data_file = os.path.join(os.path.dirname(__file__), 'solar_pv_historical_data.json')
        with open(self.history_data_file, 'r') as f:
            historical_data = json.load(f)

        P   = historical_data["power_output"] # PV power generation in Watts
        IT  = historical_data["IT"] # POA irradiation in W/m2
        Ta  = historical_data["Ta"] # ambient temperature in C
        num = historical_data["time_placeholder"] # Time of the day as an numbers between 0 and 1
        raw = historical_data["date"] # Dates as an array of strings
        i = len(Ta)

        # ******** PV Mount *******
        thetap = self.plane_tilt_angle #Plane tilt angle
        thetap = thetap * (pi / 180)
        phip = self.plane_azimuth_angle #Plane azimuth angle
        phip = phip * (pi / 180)

        # ******** Location Info ************
        _lambda = self.local_latitude #Location latitude
        Lloc = self.local_longitude #Local longitude
        Lstd = self.time_zone_longitude #Time zone longitude
        _lambda = _lambda * (pi / 180)

        # *********************************
        n = np.zeros(i) # Day number
        thetas = np.zeros(i) # Zenith angle of sun
        costhetas = np.zeros(i) # cos of zenith angle
        costhetai = np.zeros(i) # cos of incidence angle
        phis = np.zeros(i) # Azimuth of sun
        sinphis = np.zeros(i) # sin of sun azimuth angle
        delta = np.zeros(i) # Solar declination
        tstd = np.zeros(i) #Standard time
        tsol = np.zeros(i)# Solar Time
        omega = np.zeros(i) # Hour angle
        Ketta = np.zeros(i) # Incidence angle modifier
        month = np.zeros(i)#Month
        day = np.zeros(i)#Day

        x1 = np.zeros(i)
        x2 = np.zeros(i)
        x3 = np.zeros(i)
        y = np.zeros(i)

        # MIKE INDEXES
        for a in range(i):
            date_string = raw[a] # converting date to string
            m, d, _ = date_string.split('/') # Splitting the string to get month and day
            month[a] = int(m)#converting 'm' to numerical value
            day[a] = int(d)#converting 'd' to numerical value
            #calculatin n (day number)
            if month[a] == 1:
                n[a] = day[a]
            elif month[a] == 2:
                n[a] = 31 + day[a]
            elif month[a] == 3:
                n[a] = 59 + day[a]
            elif month[a] == 4:
                n[a] = 90 + day[a]
            elif month[a] == 5:
                n[a] = 120 + day[a]
            elif month[a] == 6:
                n[a] = 151 + day[a]
            elif month[a] == 7:
                n[a] = 181 + day[a]
            elif month[a] == 8:
                n[a] = 212 + day[a]
            elif month[a] == 9:
                n[a] = 243 + day[a]
            elif month[a] == 10:
                n[a] = 273 + day[a]
            elif month[a] == 11:
                n[a] = 304 + day[a]
            elif month[a] == 12:
                n[a] = 334 + day[a]

        for a in range(i):
            # We want 12AM to be 24 (not 0)
            if num[a] == 0:
                num[a] = 1

            tstd[a] = (num[a] * 24) - 0.5#hour ending data collection
            delta[a] =- sin((pi/180) * 23.45) * cos((pi/180) * 360 * (n[a] + 10) / 365.25)

            #********** Solar Time **********
            B = 360 * (n[a] - 81) / 364
            B = B * (pi / 180)
            Et = 9.87 * sin(2*B) - 7.53 * cos(B) - 1.5 * sin(B) #equation of time
            tsol[a] = tstd[a] - ((Lstd - Lloc) / 15) + (Et / 60) #solar time in hr

            #************** Hour Angle *************
            omega[a] = (tsol[a] - 12) * 15
            omega[a] = omega[a] * (pi / 180)
            costhetas[a] = cos(_lambda) * cos(delta[a]) * cos(omega[a]) + sin(_lambda) * sin(delta[a])#thetas is zenith angle of sun
            costhetas[a] = abs(costhetas[a])
            #***************************************

            thetas[a] = acos(costhetas[a]) #thetas will be in radian
            sinphis[a] =(cos(delta[a]) * sin(omega[a])) / sin(thetas[a]) #phis is azimuth of sun
            phis[a] = asin(sinphis[a])
            costhetai[a] = (sin(thetas[a]) * sin(thetap) * cos(phis[a] - phip)) + (cos(thetas[a]) * cos(thetap)) #thetai is solar incidence angle on plane
            Ketta[a] = 1 - 0.1 * ((1 / costhetai[a]) - 1) # incidence angle modifier

            #*********** Gordon and Reddy Model **********
            x1[a] = IT[a] * Ketta[a]
            x2[a] = IT[a] * Ketta[a] * Ta[a]
            x3[a] = (IT[a] * Ketta[a])**2
            y[a] = P[a]

        # Multiple Linear Regression (no intercept)
        XX = np.column_stack((x1,x2,x3))
        BB, resid, rank, s = np.linalg.lstsq(XX, y)

        return BB
