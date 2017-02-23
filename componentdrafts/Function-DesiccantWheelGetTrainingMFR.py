# coding: utf-8

import numpy as np
from scipy import stats
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import statsmodels.api as sm


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

    h_OA_fan = []

    # hot water temperature drop across regneration coil [C]
    delta_T = []

    for i in range(0, Rows):
        if Fan[i] > 0 and Vlv[i] > 0:
            h_OAx = HAPropsSI('H', 'T', (T_OA[i]+273.15), 'P', 101325, 'R', RH_OA[i]/100); #calculate outdoor air enthalpy
            hf_OA_fan.append(h_OAx)
            delta_T.append(T_hw[i] - T_hw_out[i])

    #single variabel regression based on outdoor air enthalpy
    RegressionArray = [h_OA_fan]
    x = RegressionArray
    y = delta_T

    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))

    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))

    results = sm.OLS(y, X).fit()

    Coefficients = results.params
    return Coefficients
