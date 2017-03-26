# coding: utf-8

import sqlite3
import pandas as pd
import numpy as np
import math
import numpy.matlib


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

    xs = np.column_stack((np.ones(len(x)), x))
    (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, y)
    
    return intercept, slope


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

    xs = np.column_stack((np.ones(len(X)), X))
    (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, Y)

    IR_discharge = (0 - intercept)
    InvEFFDischarge = 1 / slope
    return IR_discharge, InvEFFDischarge


def GetIdleParameters(TrainingData, Capacity):
    Time = TrainingData['Time'].values
    SOC = TrainingData['SOC'].values
    Rows = len(Time)
    PrevTime = []
    CurrTime = []
    CurrSOC = []
    PrevSOC = []
    Y = []
    X = []

    for i in range(0, Rows + 1):
        if i > 1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])

    for i in range(0, Rows - 1):
        Y.append((CurrSOC[i] - PrevSOC[i]) / ((CurrTime[i] - PrevTime[i]) * 24))
        X.append(CurrSOC[i])

    xs = np.column_stack((np.ones(len(X)), X))
    (intercept, slope), resid, rank, s = np.linalg.lstsq(xs, Y)

    return slope, intercept


def getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevSOC, InvEFF_Charge, IR_Charge, IR_DisCharge, InvEFF_DisCharge, Idle_A, Idle_B):

    # change in state of charge during this timestep. Initialization of variable    
    deltaSOC = 0

    #for batttery charging
    if InputPowerRequest > 0:
        P2 = InputPowerRequest*InvEFF_Charge-InputCurrent*InputCurrent*IR_Charge #P2 is power recieved into battery storage
        deltaSOC = P2*timestep/3600/Capacity 
        
    elif InputPowerRequest < 0:
        P2 = InputPowerRequest/InvEFF_DisCharge-InputCurrent*InputCurrent*IR_DisCharge #P2 is power released from battery storage
        deltaSOC = P2*timestep/3600/Capacity


    deltaSOC_self = (Idle_A + Idle_B * PrevSOC) * timestep / 3600
    SOC = PrevSOC + deltaSOC + deltaSOC_self
    InputPower = InputPowerRequest

    if SOC < minimumSOC:
        if PrevSOC < minimumSOC and InputPowerRequest < 0:
            InputPower = 0
            SOC = PrevSOC + deltaSOC_self
        if PrevSOC > minimumSOC and InputPowerRequest < 0:
            InputPower = InputPowerRequest * (PrevSOC - minimumSOC) / (PrevSOC - SOC)
            InputCurrent = InputCurrent * InputPower / InputPowerRequest
            P2 = InputPower / InvEFF_DisCharge - InputCurrent * InputCurrent * IR_DisCharge
            deltaSOC= P2 * timestep / 3600 / Capacity
            SOC = PrevSOC + deltaSOC + deltaSOC_self
    if SOC > 1:
            InputPower = InputPowerRequest * (1 - PrevSOC) / (SOC - PrevSOC)
            InputCurrent = InputCurrent * InputPower / InputPowerRequest
            P2 = InputPower * InvEFF_Charge - InputCurrent * InputCurrent * IR_Charge
            deltaSOC= P2 * timestep / 3600 / Capacity
            SOC = PrevSOC + deltaSOC + deltaSOC_self
    if SOC < 0:
            SOC = 0
        
    
    return SOC, InputPower


#Battery Storage Capacity in W-h
Capacity = 24000

#Input: Requested Power to battery in W.  Positive for charging, negative for discharging. 0 for idle
# Request may be truncated to an actual Input power if minimum SOC or maximum SOC (1.0) limits are reached
InputPowerRequest = -3000

#timestep in seconds
timestep = 500

#State of Charge in previous timestep; normalized 0 to 1
PrevSOC = 0.3996

# Amperes
InputCurrent = -12.5

# limit depth of discharge to prolong battery life
minimumSOC = 0.4

#This variable should toggle to control whether we re-run to get updated charging training data, or use
# exisitng/default parameters
RunTrainingCharge = True

#This variable should toggle to control whether we re-run to get updated discharging training data, or use
# exisitng/default parameters
RunTrainingDisCharge = True

#This variable should toggle to control whether we re-run to get updated idle training data, or use
# exisitng/default parameters
RunTrainingIdle = True#False

#---------------------------TRAINING FOR BATTERY CHARGING-------------------------------------------------------------------------------

if RunTrainingCharge:
    # this should reference a data file from recent battery charging ar a database of charging data.  It should only contin data from the
    # charge phase and not idle or discharge
    TrainingData = pd.read_csv('BatteryCharging.csv', header=0)
    C = GetChargingParameters(TrainingData, Capacity)

    #internal inverter efficiency from AC to DC during battery charging
    InvEFF_Charge = C[0]

    # battery internal resistance during charging, in Ohms
    IR_Charge = C[1]

else:
    # internal inverter efficiency from AC to DC during battery charging.  This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    InvEFF_Charge = 0.94

    # battery internal resistance during charging, in Ohms. This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    IR_Charge = 0.7

#--------------------------TRAINING FOR BATTERY DISCHARGING------------------------------------------------------------------------------
if RunTrainingDisCharge:
    #this should reference a data file from recent battery discharging ar a database of discharging data.  It should only contin data from the
    # discharge phase and not idle or charging
    TrainingData = pd.read_csv('BatteryDisCharging.csv', header=0)
    C = GetDisChargingParameters(TrainingData, Capacity)

    # battery internal resistance during discharging, in Ohms
    IR_DisCharge = C[0]

    #internal inverter efficiency from DC to AC during battery discharging
    InvEFF_DisCharge = C[1]
    
else:
    # internal inverter efficiency from AC to DC during battery charging.  This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    InvEFF_DisCharge = 0.94

    # battery internal resistance during charging, in Ohms. This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    IR_DisCharge = 0.7

#--------------------------TRAINING FOR BATTERY IDLE-----------------------------------------------------------------------------
if RunTrainingIdle:
    #this should
    #reference a data file from recent battery discharging ar a database of discharging data.  It should only contin data from the
    # discharge phase and not idle or charging
    TrainingData = pd.read_csv('BatteryIdle.csv', header=0)
    C = GetIdleParameters(TrainingData, Capacity)

    #constant rate of SOC loss due to battery self-discharge (change in SOC per hour)
    Idle_A = C[1]

    #rate of SOC loss multiplied by current SOC; due to battery self-discharge
    Idle_B = C[0]

else:
    # constant rate of SOC loss due to battery self-discharge (change in SOC per hour)
    Idle_A = -0.00012

    # rate of SOC loss multiplied by current SOC; due to battery self-discharge
    Idle_B = -0.00024


NewSOC, InputPower = getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevSOC, InvEFF_Charge, IR_Charge, IR_DisCharge, InvEFF_DisCharge, Idle_A, Idle_B)
print NewSOC
print InputPower
