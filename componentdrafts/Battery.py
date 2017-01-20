
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib


Capacity = 24000 #Battery Storage Capacity in W-h
InputPowerRequest = -3000 #Input: Requested Power to battery in W.  Positive for charging, negative for discharging. 0 for idle
# Request may be truncated to an actual Input power if minimum SOC or maximum SOC (1.0) limits are reached 
timestep = 500 #timestep in seconds
PrevSOC = 0.3996 #State of Charge in previous timestep; normalized 0 to 1
InputCurrent = -12.5 # Amperes
minimumSOC = 0.4 # limit depth of discharge to prolong battery life

RunTrainingCharge = 'yes' #This variable should toggle to control whether we re-run to get updated charging training data, or use 
# exisitng/default parameters
RunTrainingDisCharge = 'yes' #This variable should toggle to control whether we re-run to get updated discharging training data, or use 
# exisitng/default parameters
RunTrainingIdle = 'no' #This variable should toggle to control whether we re-run to get updated idle training data, or use 
# exisitng/default parameters

#---------------------------TRAINING FOR BATTERY CHARGING-------------------------------------------------------------------------------

if RunTrainingCharge == 'yes'or RunTrainingCharge =='Yes'or RunTrainingCharge =='YES':
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryCharging.csv', header=0) #this should
    #reference a data file from recent battery charging ar a database of charging data.  It should only contin data from the 
    # charge phase and not idle or discharge
    C = GetChargingParameters(TrainingData,Capacity)
    InvEFF_Charge=C[0] #internal inverter efficiency from AC to DC during battery charging
    IR_Charge=C[1] # battery internal resistance during charging, in Ohms
if RunTrainingCharge == 'no'or RunTrainingCharge =='No'or RunTrainingCharge =='NO':
    InvEFF_Charge=0.94 #internal inverter efficiency from AC to DC during battery charging.  This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    IR_Charge=0.7 # battery internal resistance during charging, in Ohms. This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    
#--------------------------TRAINING FOR BATTERY DISCHARGING------------------------------------------------------------------------------
if RunTrainingDisCharge == 'yes'or RunTrainingDisCharge =='Yes'or RunTrainingDisCharge =='YES':
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryDisCharging.csv', header=0) #this should
    #reference a data file from recent battery discharging ar a database of discharging data.  It should only contin data from the 
    # discharge phase and not idle or charging
    C = GetDisChargingParameters(TrainingData,Capacity)
    IR_DisCharge=C[0] # battery internal resistance during discharging, in Ohms
    InvEFF_DisCharge=C[1] #internal inverter efficiency from DC to AC during battery discharging
if RunTrainingDisCharge == 'no'or RunTrainingDisCharge =='No'or RunTrainingDisCharge =='NO':
    InvEFF_DisCharge=0.94 #internal inverter efficiency from AC to DC during battery charging.  This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training
    IR_DisCharge=0.7 # battery internal resistance during charging, in Ohms. This default value should be updated with recent
    # training results.  The value can then be re-used for a time, before re-training   

#--------------------------TRAINING FOR BATTERY IDLE-----------------------------------------------------------------------------
if RunTrainingIdle == 'yes'or RunTrainingIdle =='Yes'or RunTrainingIdle =='YES':
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryIdle.csv', header=0) #this should
    #reference a data file from recent battery discharging ar a database of discharging data.  It should only contin data from the 
    # discharge phase and not idle or charging
    C = GetIdleParameters(TrainingData,Capacity)
    Idle_A=C[1] #constant rate of SOC loss due to battery self-discharge (change in SOC per hour) 
    Idle_B=C[0] #rate of SOC loss multiplied by current SOC; due to battery self-discharge 
if RunTrainingIdle == 'no'or RunTrainingIdle =='No'or RunTrainingIdle =='NO':
    Idle_A= -0.00012 #constant rate of SOC loss due to battery self-discharge (change in SOC per hour) 
    Idle_B=-0.00024 #rate of SOC loss multiplied by current SOC; due to battery self-discharge 
    
Outputs=getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevSOC,InvEFF_Charge,IR_Charge,IR_DisCharge,InvEFF_DisCharge,Idle_A,Idle_B)

InputPower = Outputs[1]
NewSOC = Outputs[0]
print NewSOC
print InputPower
    

