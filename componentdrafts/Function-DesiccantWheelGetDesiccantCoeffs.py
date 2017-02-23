# coding: utf-8

import numpy as np
from scipy import stats
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import statsmodels.api as sm

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

    # regression variables are hot water inlet temperature, outdoor air enthalpy, outdoor air enthalpy squared and CFM of outdoor air
    RegressionArray = [T_HW_fan, h_OA_fan, h_OA2_fan, CFM_OA_fan]
    x = RegressionArray
    y = delta_h

    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()

    Coefficients = results.params
    return Coefficients
