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

from econ_dispatch.component_models import ComponentBase


class Component(ComponentBase):
    def __init__(self, current_output=10.00, **kwargs):
        super(Component, self).__init__(current_output=current_output, **kwargs)

    def get_output_metadata(self):
        return [u"electricity"]

    def get_optimization_parameters(self):
        self.predict()
        return {"current_output":self.current_output}

    def update_parameters(self, current_output=10.00):
        self.current_output = current_output

    def predict(self):
        # *********** scenario 5- DC Power, ambient temperature, and IT are measured at-site ********
        # here we assumed that regression models were built separately and
        # therefore regression coefficients are available. also, forecasted values
        # for IT and Ta are assumed to be available. This code is meant to be used
        # for 24 hours ahead predictions- The code creates an excel file named
        # "power-Output" and write the results (DC Power) on it along with date and
        # time stamps
    
        # ******** Reading Forecasted Solar and Ambient Temperatures ************
        data_file = os.path.join(os.path.dirname(__file__), 'solar_pv_dynamic_inputs.json')
        with open(data_file, 'r') as f:
            dynamic_inputs = json.load(f)

        ITf = dynamic_inputs["ITf"] # Forecasted Total solar irradiance on tilted plane or POA irradiance, [W/m2]
        Taf = dynamic_inputs["Taf"] # Forecasted ambient temperature, [Degrees C]
        TOD = dynamic_inputs["time_placeholder"] # Time of the day as a numbers between 0 and 1
        raw = dynamic_inputs["date"]# Dates as an array of strings
    
        # ******** PV Mount, User inputs *******
        theta_p = 15 #Tilt of PV panels with respect to horizontal, degrees
        theta_p = theta_p * (pi / 180) #converting degrees to radians
        phi_p = 0 # Azimuth of plane. South facing is 0 (positive for orientations west of south), degrees
        phi_p = phi_p * (pi / 180)#converting degrees to radians
        #PVMoCo = xlsread('Static-Inputs','d3') #PV mounting code: 1 for ground-mounted and 0 for roof-mounted systems
        #Rog = xlsread('Static-Inputs','e3')# Ground reflectivity
    
        # ******** Location Info, User Input OR obtain somehow from zip code or other smart means ************
        _lambda = 39.74 #Location latitude
        Lloc = -105.18 #Local longitude
    
        TimeZone = 'MST'
        if TimeZone == 'MST':
            Lstd = -105
        elif TimeZone == 'PST':
            Lstd =- 120
        elif TimeZone == 'CST':
            Lstd =- 90
        elif TimeZone == 'EST':
            Lstd =- 75
        else:
            Lstd =- 120
    
        _lambda = _lambda * (pi / 180)#converting degrees to radians
    
        # G&R model coefficients (Annual Model)
        a1, a2, a3 = self.train()
    
        # ********************************************************
        #Pac = np.zeros(24) # Inverter power output
        Pdc = np.zeros(24) # PV power output, DC
        n = np.zeros(24) # Day number
        theta_s = np.zeros(24) # Zenith angle of sun
        cos_theta_s = np.zeros(24) # cos of zenith angle
        cos_theta_i = np.zeros(24) # cos of incidence angle
        phi_s = np.zeros(24) # Azimuth of sun
        sin_phi_s = np.zeros(24) # sin of sun azimuth angle
        delta = np.zeros(24) # Solar declination
        #sindelta = np.zeros(24) # sin of solar declination
        t_std = np.zeros(24) #Standard time
        t_sol = np.zeros(24)# Solar Time
        omega = np.zeros(24) # Hour angle
        Ketta = np.zeros(24) # Incidence angle modifier
        omegaS = np.zeros(24) # sunrise and sunset hour angle
        cos_omega_S = np.zeros(24) # cos of omegaS
        daylightindicator = np.zeros(24) # indicates whether sun is above the horizon or not
        omegaSprime = np.zeros(24) # sunset and sunrise hour angle over the panel surface
        cos_omegaS_prime = np.zeros(24) # cos of omegaSprime
        daylightindicatorT = np.zeros(24)# indicates whether sun is above the edge of the panel or not (sunrise and sunset over the panel surface)
        DI = np.zeros(24)#either 0 or 1 based on daylightindicatorT
        month = np.zeros(24)#Month
        day = np.zeros(24)#Day
    
        #********* Calculating n (day number) **********
        for a in range(24): # MIKE INDEXES
            date_string = raw[a] #converting date to string
            m, d, _ = date_string.split('/')#Splitting the string to get month and day
            month[a] = int(m)#converting 'm' to numerical value
            day[a] = int(d)#converting 'd' to numerical value
    
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
    
        # MIKE INDEXES
        for a in range(24):
            t_std[a] = (TOD[a] * 24) - 0.5 # hour ending data collection
            delta[a] = -sin((pi/180)*23.45) * cos((pi/180)*360*(n[a]+10)/365.25)
    
            # Solar Time
            B = 360 * (n[a] - 81) / 364
            B = B * (pi / 180)
            Et = 9.87 * sin(2*B) - 7.53 * cos(B) - 1.5 * sin(B) # equation of time
            t_sol[a] = t_std[a] - ((Lstd - Lloc) / 15) + (Et / 60) # solar time in hr
    
            # Hour Angle
            omega[a] = (t_sol[a] - 12) * 15
            omega[a] = omega[a] * (pi / 180)
            cos_theta_s[a] = cos(_lambda) * cos(delta[a]) * cos(omega[a]) + sin(_lambda) * sin(delta[a]) # thetas is zenith angle of sun
            sin_theta_s = sin(pi/2-acos(cos_theta_s[a]))
            cos_theta_s[a] = abs(cos_theta_s[a])
    
            # Sunrise and sunset solar angle for horizontal surfaces
            if sin_theta_s<0:
                DI[a]=0
            else:
                DI[a]=1
    
            # **************************************
            theta_s[a] = acos(cos_theta_s[a]) #thetas will be in radian
            sin_phi_s[a] = (cos(delta[a]) * sin(omega[a])) / sin(theta_s[a]) #phis is azimuth of sun
            phi_s[a] = asin(sin_phi_s[a])
            cos_theta_i[a] = (sin(theta_s[a]) * sin(theta_p) * cos(phi_s[a] - phi_p)) + (cos(theta_s[a]) * cos(theta_p)) # thetai is solar incidence angle on plane
            Ketta[a] = 1 - 0.1 * ((1 / cos_theta_i[a]) - 1) # incidence angle modifier
    
            # Power Calculation- G&R model
            Pdc[a] = DI[a] * (a1 * ITf[a] * Ketta[a] + a2 * ITf[a] * Ketta[a] * Taf[a] + a3 * (ITf[a] * Ketta[a])**2)
    
    
        return Pdc
    
    def train(self):
        # This module reads the historical data on ambient temperatures (in C), PV power
        # generation (in W) and POA irradiation (in W/m2); then, calculates the
        # ketta using time stamps of the historical data and fit the power prediction model.
        # At the end, regression coefficients will be written to a file.

        data_file = os.path.join(os.path.dirname(__file__), 'solar_pv_historical_data.json')
        with open(data_file, 'r') as f:
            historical_data = json.load(f)
        
        P   = historical_data["power_output"] # PV power generation in Watts
        IT  = historical_data["IT"] # POA irradiation in W/m2
        Ta  = historical_data["Ta"] # ambient temperature in C
        num = historical_data["time_placeholder"] # Time of the day as an numbers between 0 and 1
        raw = historical_data["date"] # Dates as an array of strings
        i = len(Ta)

        data_file = os.path.join(os.path.dirname(__file__), 'solar_pv_static_inputs.json')
        with open(data_file, 'r') as f:
            static_inputs = json.load(f)
    
        # ******** PV Mount *******
        thetap = static_inputs["plane_tilt_angle"] #Plane tilt angle
        thetap = thetap * (pi / 180)
        phip = static_inputs["plane_azimuth_angle"] #Plane azimuth angle
        phip = phip * (pi / 180)
    
        # ******** Location Info ************
        _lambda = static_inputs["local_latitude"] #Location latitude
        Lloc = static_inputs["local_longitude"] #Local longitude
        Lstd = static_inputs["time_zone_longitude"] #Time zone longitude
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
