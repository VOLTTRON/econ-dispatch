# coding: utf-8

import sqlite3
import pandas as pd
import numpy as np
import math
import numpy.matlib

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
RunTrainingIdle = False

#---------------------------TRAINING FOR BATTERY CHARGING-------------------------------------------------------------------------------

if RunTrainingCharge:
    # this should reference a data file from recent battery charging ar a database of charging data.  It should only contin data from the
    # charge phase and not idle or discharge
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryCharging.csv', header=0)
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
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryDisCharging.csv', header=0)
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
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryIdle.csv', header=0)
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
