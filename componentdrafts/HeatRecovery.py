
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

MFR_ex= 2 #kg/s
MFR_water= 15 #kg/s
T_ex_i= 311 #sensor input: exhaust fluid temperature inlet to heat recovery in degrees C
T_water_i=60 #sensor input: building hot water loop fluid temperature inlet to heat recovery in degrees C
CurrSpeed=0.75 # sensor input: current speed of CHP prime mover.  Used only for regression.  Input units are not important and can be a percent speed or a fuel input parameter

TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\MicroturbineData.csv', header=0)

PrimeMoverFluid = 'FlueGas'
HotWaterLoopFluid = 'Water'
HeatExchangerArea = 8 #m2; default.  Only used if Method =1
DefaultEffectiveness=0.9 #Only used if Method =2
C=[0,0]

Method=3 # Method 1 calls for input of the heat transfer area - which is sometimes found on spec sheets for HXs - then calculates the effectiveness based on NTU method
#Method 2 is a simplified approach that calls for the user to specify a default heat exchanger effectiveness
# Method 3 builds a regression equation to calculate the outlet water temperature based on training data
NeedRegressionCoefficients='yes' #Training data may only need to be used periodically to find regression coefficients, which can thereafter be re-used
if NeedRegressionCoefficients=='yes'and Method ==3:
    C = GetRegressionHeatRecovery(TrainingData)

TWaterOut = HeatRecoveryWaterOut(PrimeMoverFluid,HotWaterLoopFluid,Method,T_water_i,T_ex_i,MFR_water,MFR_ex,CurrSpeed,C,HeatExchangerArea, DefaultEffectiveness)
HeatRecovered = GetHeatRecovered(HotWaterLoopFluid,T_water_i,TWaterOut,MFR_water)
print TWaterOut
print HeatRecovered 

