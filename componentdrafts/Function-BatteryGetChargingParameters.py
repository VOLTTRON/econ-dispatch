# coding: utf-8

from scipy import stats
import numpy as np

def GetChargingParameters(TrainingData, Capacity):

    Time = TrainingData['Time'].values
    Current = TrainingData['I'].values
    PowerIn = TrainingData['Po'].values
    SOC = TrainingData['SOC'].values
    x = len(Time)
    PrevTime = []
    CurrTime = []
    CurrPower = []
    PrevPower = []
    CurrSOC = []
    PrevSOC = []
    CurrentI = []
    ChargeEff = []
    Slope_RI = []

    for i in range(0, x + 1):
        if i>1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrPower.append(PowerIn[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
            CurrentI.append(Current[i-1])

    for i in range(0,x-1):
        ChargeEff.append((CurrSOC[i] - PrevSOC[i]) * Capacity / ((CurrTime[i] - PrevTime[i]) * 24) / CurrPower[i])
        Slope_RI.append(0 - CurrentI[i] * CurrentI[i] / CurrPower[i])

    x = Slope_RI
    y = ChargeEff
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return intercept, slope
