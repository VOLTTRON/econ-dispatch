import json
import numpy as np



def FuelCell_Operate(Power, Tin, Starts, NetHours, Coef):
    F = 96485 #Faraday constant C/mol

    if Coef.Fuel == "CH4":
        n = 8 # of electrons per molecule (assuming conversion to H2)
        LHV = 50144 #Lower heating value of CH4 in kJ/g
        m_fuel = 16#molar mass
    elif Coef.Fuel == "H2":
        n = 2 # of electrons per molecule (assuming conversion to H2)
        LHV = 120210 #Lower heating value of H2 in kJ/kmol
        m_fuel = 2#molar mass
    else:
        raise ValueError("Unknown Fuel")

    nPower = Power / Coef.NominalPower
    ASR = Coef.NominalASR + Coef.ReStartDegradation * Starts + Coef.LinearDegradation * max(0, (NetHours - Coef.ThresholdDegradation)) #ASR  in Ohm*cm^2
    Utilization = Coef.Utilization[0] * (1 - nPower)**2 + Coef.Utilization[1] * (1 - nPower) + Coef.Utilization[2] #decrease in utilization at part load
    Current = Coef.Area * (Coef.NominalCurrent[0] * nPower**2 + Coef.NominalCurrent[1] * nPower + Coef.NominalCurrent[2]) #first guess of current
    HeatLoss = Power * Coef.StackHeatLoss
    AncillaryPower = 0.1 * Power

    for j in range(4):
        Voltage = Coef.Cells * (Coef.NominalOCV - Current * ASR / Coef.Area)
        Current = Coef.gain * (Power + AncillaryPower) * 1000 / Voltage - (Coef.gain -1 ) * Current
        FuelFlow = m_fuel * Coef.Cells * Current / (n * 1000 * F * Utilization)
        ExhaustFlow = (Coef.Cells * Current * (1.2532 - Voltage / Coef.Cells) / 1000 - HeatLoss) / (1.144 * Coef.StackDeltaT) #flow rate in kg/s with a specific heat of 1.144kJ/kg*K
        AncillaryPower = Coef.AncillaryPower[0] * FuelFlow**2  + Coef.AncillaryPower[1] * FuelFlow + Coef.AncillaryPower[0] * ExhaustFlow**2 + Coef.AncillaryPower[1]*ExhaustFlow + Coef.AncillaryPower[2] * (Tin - 18) * ExhaustFlow

    ExhaustTemperature = ((Coef.Cells * Current *(1.2532 - Voltage / Coef.Cells) / 1000 - HeatLoss) + (1 - Utilization) * FuelFlow * LHV) / (1.144 * ExhaustFlow) + Tin + (Coef.ExhaustTemperature[0]*nPower**2 + Coef.ExhaustTemperature[1] * nPower + Coef.ExhaustTemperature[2])
    NetEfficiency = Power / (FuelFlow * LHV)
    
    return FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency


class Coefs(object):
    pass

Coef = Coefs()
Coef.Fuel = 'CH4'
Coef.ThresholdDegradation = float(8e3) #hours before which there is no linear degradation
Coef.NominalCurrent = np.array([0.1921, 0.1582, 0.0261])
Coef.NominalASR = 0.5
Coef.ReStartDegradation = float(1e-3)
Coef.LinearDegradation = float(4.5e-6)
Coef.ThresholdDegradation = float(9e3)#hours before which there is no linear degradation
Coef.Utilization = np.array([-.25, -.2, .65])
Coef.StackHeatLoss = 0.1
Coef.AncillaryPower = np.array([0.5, 4.0, 0.25])
Coef.Area = 5000.0 #5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
Coef.StackDeltaT = 100.0
Coef.ExhaustTemperature = np.array([0.0, 0.0, 0.0])
Coef.gain = 1.4


NominalV = np.array([0.775, 0.8, 0.814])
for i in range(3):
    with open("gen{}.json".format(i), 'r') as f:
        gen = json.load(f)

    Coef.NominalPower = gen["NominalPower"]
    Coef.Cells = gen["NominalPower"] #currently set up for 1kW cells
    Coef.NominalOCV = NominalV[i]
    
    GenPower = 300.0
    T_amb = 20.0
    GenStart = 5.0
    GenHours = 7000.0
    x = FuelCell_Operate(GenPower, T_amb, GenStart, GenHours, Coef)

    print FuelCell_Operate(GenPower, T_amb, GenStart, GenHours, Coef)
    FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency = FuelCell_Operate(GenPower, T_amb, GenStart, GenHours, Coef)
