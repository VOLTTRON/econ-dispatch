# coding: utf-8

#Desiccant Wheel: Main Program
import sqlite3
import pandas as pd
import numpy as np
import math
import numpy.matlib
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI


def getDesiccantCoeffs(TrainingData,h_oa_min):

    #This function is used by th Desiccant Wheel program and uses historical trend data on outdoor and conditioned air temperuatures and relative humidities, outdoor air flow rates, fan status and outdoor air temperature
    # A regression dequation is built to predict the drop in enthalpy of the supply air acrsoss the desiccant coil as a function of outdoor air enthalpy, outdoor enthalpy squared, hot water temperature and CFM of outdoor air

    #timestamp
    Time = TrainingData['Date/Time'].values

    # Outdoor air temperature sensor; convert to degrees C
    T_OA = TrainingData['T_OA'].values

    # Outdoor air relative humudity (fraction)
    RH_OA = TrainingData['RH_OA'].values

    # Conditioned temperature of supply air at the outlet of the heat recovery coil
    T_CA = TrainingData['T_CA'].values
    
    # Volumetric flow rate of outdor air in cubic feet per minute (cfm)
    CFM_OA = TrainingData['CFM_OA'].values

    # Conditioned relative humidity of supply air at the outlet of the heat recovery coil
    RH_CA = TrainingData['RH_CA'].values

    # hot water inlet temperature to regeneration coil
    T_hw = TrainingData['T_HW'].values

    # Fan status (1=ON, 0=OFF)
    Fan = TrainingData['FanStatus'].values

    Rows = len(Time)

    h_OA = []
    h_CA = []
    delta_h = []
    h_OA_fan = []
    h_OA2_fan = []
    T_HW_fan = []
    CFM_OA_fan = []

    for i in range(0, Rows):
        if Fan[i] > 0:
            h_OAx = HAPropsSI('H','T', (T_OA[i]+273.15), 'P', 101325, 'R', RH_OA[i]/100) #calculate outdoor air enthalpy
            if  h_OAx > h_oa_min:
                h_OA.append(h_OAx)
                h_CAx = HAPropsSI('H', 'T', (T_CA[i]+273.15), 'P', 101325, 'R', RH_CA[i]/100)
                h_CA.append(h_CAx)
                delta_h.append(h_CAx-h_OAx)
                h_OA_fan.append(h_OAx)
                h_OA2_fan.append(pow(h_OAx, 2))
                T_HW_fan.append(T_hw[i])
                CFM_OA_fan.append(CFM_OA[i])


    y = delta_h
    ones = np.ones(len(T_HW_fan))
    X = np.column_stack((CFM_OA_fan, h_OA2_fan, h_OA_fan, T_HW_fan, ones))
    Coefficients, resid, rank, s = np.linalg.lstsq(X, y)

    return Coefficients


def getTrainingMFR(TrainingData):
    #This function is used by the Desiccant Wheel program to estimate the delta T across the regeneration coil for the purpose of calculating the hot water mass flow rate in the regeneration coil
    # This funciton is used only when historical data for the regneration coil outlet tempreature is available.

    # Outdoor air temperature sensor; convert to degrees C
    T_OA = TrainingData['T_OA'].values

    # Outdoor air relative humudity (fraction)
    RH_OA = TrainingData['RH_OA'].values

    # regeneration coil valve command (fraction)
    Vlv = TrainingData['Regen_Valve'].values

    # Hot water inlet temperature to regeneration coil [C]
    T_hw = TrainingData['T_HW'].values

    # Hot water outlet temperature from regeneration coil [C]
    T_hw_out = TrainingData['T_HW_out'].values

    # Fan status (1=ON, 0=OFF)
    Fan = TrainingData['FanStatus'].values

    Rows = len(T_hw_out)

    h_OA_fan = []

    # hot water temperature drop across regneration coil [C]
    delta_T = []

    for i in range(0, Rows):
        if Fan[i] > 0 and Vlv[i] > 0:
            h_OAx = HAPropsSI('H', 'T', (T_OA[i]+273.15), 'P', 101325, 'R', RH_OA[i]/100); #calculate outdoor air enthalpy
            h_OA_fan.append(h_OAx)
            delta_T.append(T_hw[i] - T_hw_out[i])

    #single variabel regression based on outdoor air enthalpy
    y = delta_T
    ones = np.ones(len(h_OA_fan))
    X = np.column_stack((h_OA_fan, ones))
    Coefficients, resid, rank, s = np.linalg.lstsq(X, y)

    return Coefficients


GetTraining = True

# An optional input is the hot water outlet temperature from the regeneration coil.
# This data is used only in a training function to provide a better estimate of the detla_T in the coil, compared to default assumption of 5 degrees C.
# This training also requires an input of the hot water coil valve command, which is used as a filter of the data for building the regression
T_HW_out_available = True

### DEFAULT ASSUMPTIONS
# Electric power in Watts consumed to run desiccant system, typically including desiccant wheel motor and heat recovery wheel motor
WheelPower = 300

# Assumed HW temperature drop across Regeneration Coil, if HW outlet temperature is not measured
deltaT_RegenCoil = 5

# specific heat of water in J/g-C
cp_water = 4.186

### REQUIRED INPUTS
# CFM of outdoor air.  This is expected to be a constant value when the fan is on (minimum outdoor air flow rate)
cfm_OA = 1000

# Fan status for air-hanling unit with desiccant system (1= ON, 0 = OFF)
Fan_Status = 1

# outdoor air temperature, from building sensor, in degrees C
T_OA = 32

# outdoor air relative humidity, from building sensor in kg/m3
RH_OA = 0.6

# Hot water loop temperature in degrees C
T_HW = 46.1


if GetTraining:
    TrainingData = pd.read_csv('SampleDesiccant.csv', header=0)
    w_oa_min = HAPropsSI('W', 'T', (11.67+273.15), 'P', 101325, 'R', 1) #This is an estimate of the minimum humidy ratio down to which dehumidifcation is useful in relieving cooling coil latent cooling loads.  The assumed humidity conditions are saturation (100% RH) at 53 degrees F, which is a typical cooling coil setpoint for dehumidification
    h_oa_min = HAPropsSI('H', 'T', (11.67+273.15), 'P', 101325, 'R', 1) #corresponding minimum enthalpy
    print h_oa_min
    Coefs = getDesiccantCoeffs(TrainingData, h_oa_min) # regression is a funciton of MODEL variables, rather than AHU system variables, since the AHU system is not modeled.  specific regression variables are outdoor air enthalpy, outdoor air enthalpy squared, hot water inlet temperature, and outdoor air CFM (which may be a constant value when the fan is on)

    # If hot water outlet temperature from regeneration coil is available, use historical data on hot water coil inlet and outlet temperatures along with valve command to build regression.
    if T_HW_out_available:
        Coefs2 = getTrainingMFR(TrainingData)


# outdoor air enthalpy in current timestep
h_oa_curr = HAPropsSI('H', 'T', (T_OA+273.15), 'P', 101325, 'R', RH_OA)

# outdoor air humidity ratio in current timestep
w_oa_curr = HAPropsSI('W', 'T', (T_OA+273.15), 'P', 101325, 'R', RH_OA)

# If hot water outlet temperature from regeneration coil is available, get estimate of delta T as a funciton of outdoor air enthalpy.
if T_HW_out_available:
    deltaT_RegenCoil = Coefs2[0] * h_oa_curr + Coefs2[1]

# Thermal COP is the reduction in cooling load divided by the heat input to the desiccant wheel. The regression equation here is based on Figure 6 in http://www.nrel.gov/docs/fy05osti/36974.pdf
COP_thermal = 0.2482 + 4.171529 * w_oa_curr - 0.00019 * T_OA + 0.000115 * pow(T_OA, 2)

# Calculate the change in enthalpy from the outdoor air to the conditioned air stream, prior to mixing box (or cooling coil if it is a 100% OA unit)
deltaEnthalpy = Coefs[4] + T_HW * Coefs[3] + h_oa_curr * Coefs[2] + pow(h_oa_curr,2) * Coefs[1] + cfm_OA * Coefs[0]

# No useful enthalpy reduction if outdoor air humidity ratio is already below saturated conditions at the cooling coil outlet temperature used for dehumidification
if w_oa_curr < w_oa_min or Fan_Status == 0: 
    deltaEnthalpy = 0

# Useful enthalpy reduction should be limited by enthalpy of the supply air at saturation for the cooling coil in dehumidification mode
elif deltaEnthalpy < (h_oa_min - h_oa_curr):
    deltaEnthalpy = h_oa_min - h_oa_curr


# density of outdoor air in kg/m3 as a function of outdoor air temperature in degrees C
rho_OA = 0.0000175 * pow(T_OA, 2) - (0.00487 * T_OA) + 1.293

# calculate mass flow rate of outdoor air, given volumetric flow rate and air density
MFR_OA= cfm_OA * rho_OA / pow(3.28, 3) / 60

# output: reduction in cooling load (W)
Q_Cool_Out = -deltaEnthalpy * MFR_OA

# output: hot water loop heat input (W)
Q_HW_In = Q_Cool_Out / COP_thermal

#output: mass flow rate of water in heating coil [kg/s]
MFR_HW_In = Q_HW_In / cp_water / deltaT_RegenCoil / 1000

#output: hot water temperature outlet from regeneration coil [C]
T_HW_Out= T_HW-deltaT_RegenCoil*Fan_Status

#output: Electricity Input [W]
Q_Elec_In = WheelPower
if Q_Cool_Out == 0:
    Q_Elec_In = 0

print Q_Cool_Out
print Q_HW_In
print Q_Elec_In
print MFR_HW_In
print T_HW_Out
