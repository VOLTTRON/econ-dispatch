
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BatteryCost=15000  #replacement battery cost in $
DODCurve = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryDODcurve.csv', header=0)
SOCForecast = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\Battery24hrSOCforecast.csv', header=0)
DOD= DODCurve['DOD'].values
Cycles=DODCurve['Cycles'].values
SOC = SOCForecast['SOC_Forecast'].values
Hour = SOCForecast['Hour'].values
Depreciation =0

Rows=len(SOC)

PrevSOC = []
CurrSOC = []
for i in range(0,Rows+1):
    if i>1:
        PrevSOC.append(SOC[i-2])
        CurrSOC.append(SOC[i-1])

for i in range(0,Rows-1): 
    if i==0:
        Peak=PrevSOC[i]
        Valley=PrevSOC[i]
    if CurrSOC[i]<PrevSOC[i]:
        Valley=CurrSOC[i]
    if CurrSOC[i]>PrevSOC[i] and PrevSOC[i]==Valley and i>1:
        CurrDOD=Peak-Valley
        print CurrDOD
        CurrCycles= np.interp(CurrDOD, DOD, Cycles)
        print CurrCycles
        CurrDepreciation =  1/CurrCycles*BatteryCost
        print CurrDepreciation
        Depreciation=Depreciation+CurrDepreciation
    if CurrSOC[i]>PrevSOC[i]:
        Peak=CurrSOC[i]


print Depreciation

