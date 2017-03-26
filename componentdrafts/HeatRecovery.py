# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import math


def GetHeatRecovered(HotWaterLoopFluid,T_water_i,TWaterOut,MFR_water):
    if HotWaterLoopFluid == 'Water':
        cp_water = 4186 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater30':
        cp_water = 3913 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater50':
        cp_water = 3558 #j/kg-K
    else:
        cp_water = 4186 #j/kg-K

    HeatRecovered = MFR_water * cp_water * (TWaterOut - T_water_i) / 1000 #kW
    return HeatRecovered


def GetRegressionHeatRecovery(TrainingData):

    Twi = TrainingData['T_wi'].values
    Two = TrainingData['T_wo'].values
    Texi = TrainingData['T_ex_i'].values
    Speed = TrainingData['Speed'].values
    MFRw = TrainingData['MFR_water'].values
    Proportional = Speed / MFRw * (Texi - Twi)
    Output = Two - Twi

    x = Proportional
    y = Output

    xs = np.column_stack((np.ones(len(x)), x))
    (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, y)

    return intercept, slope


def HeatRecoveryWaterOut(PrimeMoverFluid, HotWaterFluid, Method, T_water_i, T_ex_i, MFR_water, MFR_ex, CurrSpeed, TrainingData, HeatExchagerArea, DefaultEffectiveness):

    if PrimeMoverFluid == 'FlueGas':
        cp_ex=1002 #j/kg-K
    elif PrimeMoverFluid == 'Water':
        cp_ex=4186 #j/kg-K
    elif PrimeMoverFluid == 'GlycolWater30':
        cp_ex=3913 #j/kg-K
    elif PrimeMoverFluid == 'GlycolWater50':
        cp_ex=3558 #j/kg-K
    else:
        cp_ex=4186 #j/kg-K

    if HotWaterLoopFluid == 'Water':
        cp_water=4186 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater30':
        cp_water=3913 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater50':
        cp_water=3558 #j/kg-K
    else:
        cp_water=4186 #j/kg-K

    C_ex = MFR_ex * cp_ex
    C_water = MFR_water * cp_water
    C_min = min(C_ex, C_water)
    C_max = max(C_ex, C_water)
    C_r = C_min / C_max

    if Method == 1:
        U = 650 #w/m2-K
        UA = U * HeatExchagerArea #W-K
        NTU = UA / C_min
        Eff = (1 - math.exp(-1 * NTU * (1 - C_r))) / (1 - C_r * math.exp(-1 * NTU * (1 - C_r))) #relation for concentric tube in counterflow
        T_water_o = T_water_i + Eff * C_min / C_water * (T_ex_i - T_water_i)
    elif Method == 2:
        Eff = DefaultEffectiveness
        T_water_o = T_water_i + Eff * C_min / C_water * (T_ex_i - T_water_i)
    elif Method == 3:
        T_water_o = C[0] + T_water_i + C[1] * CurrSpeed / MFR_water * (T_ex_i - T_water_i)

    return T_water_o

#kg/s
MFR_ex = 2

#kg/s
MFR_water = 15

#sensor input: exhaust fluid temperature inlet to heat recovery in degrees C
T_ex_i = 311

#sensor input: building hot water loop fluid temperature inlet to heat recovery in degrees C
T_water_i = 60

# sensor input: current speed of CHP prime mover.  Used only for regression.  Input units are not important and can be a percent speed or a fuel input parameter
CurrSpeed = 0.75

TrainingData = pd.read_csv('MicroturbineData.csv',  header=0)

PrimeMoverFluid = 'FlueGas'
HotWaterLoopFluid = 'Water'

#m2; default.  Only used if Method =1
HeatExchangerArea = 8

#Only used if Method =2
DefaultEffectiveness = 0.9

C = [0, 0]

# Method 1 calls for input of the heat transfer area - which is sometimes found on spec sheets for HXs - then calculates the effectiveness based on NTU method
# Method 2 is a simplified approach that calls for the user to specify a default heat exchanger effectiveness
# Method 3 builds a regression equation to calculate the outlet water temperature based on training data
Method = 3

#Training data may only need to be used periodically to find regression coefficients,  which can thereafter be re-used
NeedRegressionCoefficients = True
if NeedRegressionCoefficients and Method == 3:
    C = GetRegressionHeatRecovery(TrainingData)

TWaterOut = HeatRecoveryWaterOut(PrimeMoverFluid, HotWaterLoopFluid, Method, T_water_i, T_ex_i, MFR_water, MFR_ex, CurrSpeed, C, HeatExchangerArea,  DefaultEffectiveness)
HeatRecovered = GetHeatRecovered(HotWaterLoopFluid, T_water_i, TWaterOut, MFR_water)
print TWaterOut
print HeatRecovered
