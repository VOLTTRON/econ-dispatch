import numpy as np
import json
from math import ceil, isnan


def GasTurbine_Calibrate(Coef, Time, Power, Temperature, FuelFlow, AirFlow, ExhaustTemperature):
    #This function determines the best fit coefficients for modeling a gas turbine
    ## User can provide the following Coeficients, or they can be calculated here:
    # 'NominalPower': The nominal power of the turbine
    # 'Fuel_LHV': Lower heating value of the fuel in kJ/kg
    # 'TempDerateThreshold': The temperature (C) at which the turbine efficiency starts to decline
    # 'TempDerate': The rate at which efficiency declines (#/C) after the threshold temperature
    # 'Maintenance': The rate of performance decline between maintenance cycles (#/yr)
    ## Other inputs are:
    # Time (cumulative hours of operation)
    # Power (kW)
    # Temperature (C) ambient
    # FuelFlow (kg/s)
    # AirFlow (kg/s) (Optional)
    # Exhaust Temperature (C)  (Optional)

    try:
        Coef.NominalPower
    except AttributeError:
        P2 = sorted(Power)
        Coef.NominalPower = P2[int(ceil(0.98 * len(Power)))]

    try:
        Coef.Fuel_LHV
    except AttributeError:
        Coef.Fuel_LHV = 50144 # Assume natural gas with Lower heating value of CH4 in kJ/kg

    Eff = Power / (FuelFlow * Coef.Fuel_LHV)
    Eff = np.nan_to_num(Eff) # zero out any bad FuelFlow data

    # Much of this block may need reevaluation.
    # It doesn't work with with the available inputs
    try:
        Coef.TempDerate
    except AttributeError:
        Tsort = sorted(Temperature)

        # The assignments may not be the min and max of the sorted array
        Tmin = Tsort[int(ceil(0.1 * len(Tsort)))]
        Tmax = Tsort[int(ceil(0.9 * len(Tsort)))]
        C = np.zeros(10)
        # 'valid' can be all False values causing np.polyfit to fail
        for i in range(10):
            # minus one might need to be changed for python zero indexing?
            tl = Tmin + (i - 1) / 10 * (Tmax - Tmin)
            th = Tmin + i / 10 * (Tmax - Tmin)
            valid = (Eff > 0) * (Eff < 0.5) * (Temperature > tl) * (Temperature < th)
            fit = np.polyfit(Temperature[valid], Eff[valid]*100, 1)
            C[i] = -fit[1]

        #negligable dependence on temperature
        C[C < 0.025] = 0

        try:
            I = [c for c in C if c > 0.1][0]
            Coef.TempDerateThreshold = Tmin + I / 10 * (Tmax - Tmin)
            Coef.TempDerate = np.mean(C[C>0])
        except IndexError:
            Coef.TempDerateThreshold = 20
            Coef.TempDerate = 0

    # This will not work if time is not passed
    try:
        Coef.Maintenance
    except AttributeError:
        valid = (Eff > 0) * (Eff < 0.5)
        fit = np.polyfit(Time[valid]/8760, Eff[valid]*100, 1) #time in yrs, eff in %
        Coef.Maintenance = -fit(1)

    Tderate = Coef.TempDerate * (Temperature - Coef.TempDerateThreshold)
    Tderate[Tderate < 0] = 0

    if Time.size != 0:
        MaintenanceDerate = Time / 8760.0 * Coef.Maintenance
    else:
        MaintenanceDerate = 0

    # remove temperature dependence from data prior to fit
    Eff = Eff + Tderate / 100 + MaintenanceDerate / 100

    # efficiency of gas turbine before generator conversion
    Coef.Eff = np.polyfit(Power/Coef.NominalPower, Eff, 2)

    if AirFlow.size != 0:
        Coef.FlowOut = np.polyfit(Power / Coef.NominalPower, AirFlow / FuelFlow, 1)
    elif ExhaustTemperature: #assume a heat loss and calculate air mass flow
        AirFlow = (FuelFlow * Coef.Fuel_LHV - 1.667 * Power) / (1.1 * (ExhaustTemperature - Temperature))
        Coef.FlowOut = np.polyfit(Power / Coef.NominalPower, AirFlow/FuelFlow, 1)
    else:
        # assume a flow rate
        Coef.FlowOut = 1

    if ExhaustTemperature != []:
        # flow rate in kg/s with a specific heat of 1.1kJ/kg*K
        Coef.HeatLoss =  np.mean(((FuelFlow * Coef.Fuel_LHV - Power) - (ExhaustTemperature - Temperature) * 1.1 * AirFlow) / Power)
    else:
        Coef.HeatLoss = 2.0 / 3.0

    return Coef


def GasTurbine_Operate(Power, Tin, NetHours, Coef):
    #Pdemand in kW
    #Tin in C
    #Net Hours in hours since last maintenance event
    Tderate = Coef.TempDerate / 100.0 * (Tin - Coef.TempDerateThreshold)
    Tderate[Tderate < 0] = 0
    MaintenanceDerate = NetHours / 8760.0 * Coef.Maintenance / 100#efficiency scales linearly with hours since last maintenance.
    Pnorm = Power / Coef.NominalPower

    Efficiency = (Coef.Eff[0] * Pnorm**2 + Coef.Eff[1] * Pnorm + Coef.Eff[2]) - Tderate - MaintenanceDerate
    FuelFlow = Power / (Efficiency * Coef.Fuel_LHV)
    AirFlow = FuelFlow * (Coef.FlowOut[0] * Pnorm + Coef.FlowOut[1]) #air mass flow rate in kg/s
    Tout = Tin + (FuelFlow * Coef.Fuel_LHV - (1 + Coef.HeatLoss) * Power) / (1.1 * AirFlow) #flow rate in kg/s with a specific heat of 1.1kJ/kg*K

    return AirFlow, FuelFlow, Tout, Efficiency


class Coefs(object):
    pass


def main():
    #Coeficients are for a Capstone C60 gas turbine
    Coef = Coefs()
    Coef.NominalPower = 60.0
    Coef.Fuel_LHV = 50144.0
    Coef.TempDerateThreshold = 15.556 #from product specification sheet
    Coef.TempDerate = 0.12 # (#/C) from product specification sheet
    Coef.Maintenance = 3.0 # in percent/year
    Coef.HeatLoss = 2.0 / 3.0
    Coef.Eff = np.array([-0.2065, 0.3793, 0.1043])
    Coef.FlowOut = np.array([-65.85, 164.5])

    ## illustrates the calibration of the GasTurbine function
    with open('CapstoneTurndownData.json', 'r') as f:
        capstone_turndown_data = json.load(f)

    Pdemand = np.array(capstone_turndown_data['Pdemand'])
    Temperature = np.array(capstone_turndown_data['Temperature']) - 273
    FuelFlow = np.array(capstone_turndown_data['FuelFlow'])
    AirFlow = np.array(capstone_turndown_data['AirFlow'])
    Time = np.array([])
    Coef = GasTurbine_Calibrate(Coef, Time, Pdemand, Temperature, FuelFlow, AirFlow, [])
    ###---###

    AirFlow, FuelFlow, Tout, Efficiency = GasTurbine_Operate(Pdemand, Temperature, 0, Coef)
    for z in zip(AirFlow, FuelFlow, Tout, Efficiency):
        print z


if __name__ == '__main__':
    main()
