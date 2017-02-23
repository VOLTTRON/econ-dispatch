# coding: utf-8


from scipy import stats
import numpy as np

def GetDisChargingParameters(TrainingData, Capacity):

    Time = TrainingData['Time'].values
    Current = TrainingData['I'].values
    PowerIn = TrainingData['Po'].values
    SOC = TrainingData['SOC'].values
    Rows = len(Time)
    PrevTime = []
    CurrTime = []
    CurrPower= []
    PrevPower = []
    CurrSOC = []
    PrevSOC = []
    CurrentI = []
    Y = []
    X = []

    for i in range(0, Rows + 1):
        if i > 1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrPower.append(PowerIn[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
            CurrentI.append(Current[i-1])

    for i in range(0,Rows-1):
        Y.append((CurrSOC[i] - PrevSOC[i]) * Capacity / ((CurrTime[i] - PrevTime[i]) * 24) / (CurrentI[i] * CurrentI[i]))
        X.append(CurrPower[i] / (CurrentI[i] * CurrentI[i]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    IR_discharge = (0 - intercept)
    InvEFFDischarge = 1 / slope
    return IR_discharge, InvEFFDischarge
