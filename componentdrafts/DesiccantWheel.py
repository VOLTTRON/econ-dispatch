
# coding: utf-8

# In[ ]:

#Desiccant Wheel: Main Program
import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI
import statsmodels.api as sm

GetTraining = 'yes' 
T_HW_out_available='yes' # An optional input is the hot water outlet temperature from the regeneration coil.  This data is used only in a training function to provide a better estimate of the detla_T in the coil, compared to default assumption of 5 degrees C.  This training also requires an input of the hot water coil valve command, which is used as a filter of the data for building the regression

# DEFAULT ASSUMPTIONS
WheelPower = 300 #Electric power in Watts consumed to run desiccant system, typically including desiccant wheel motor and heat recovery wheel motor
deltaT_RegenCoil=5 # Assumed HW temperature drop across Regeneration Coil, if HW outlet temperature is not measured
cp_water = 4.186 #specific heat of water in J/g-C

# Required Inputs 
cfm_OA=1000 # CFM of outdoor air.  This is expected to be a constant value when the fan is on (minimum outdoor air flow rate)
Fan_Status=1 #Fan status for air-hanling unit with desiccant system (1= ON, 0 = OFF)
T_OA = 32 # outdoor air temperature, from building sensor, in degrees C
RH_OA = 0.6 # outdoor air relative humidity, from building sensor in kg/m3
T_HW = 46.1 # Hot water loop temperature in degrees C



if GetTraining == 'yes'or GetTraining =='Yes'or GetTraining =='YES':

    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\SampleDesiccant.csv', header=0) 
    w_oa_min = HAPropsSI('W','T',(11.67+273.15),'P',101325,'R',1); #This is an estimate of the minimum humidy ratio down to which dehumidifcation is useful in relieving cooling coil latent cooling loads.  The assumed humidity conditions are saturation (100% RH) at 53 degrees F, which is a typical cooling coil setpoint for dehumidification
    h_oa_min = HAPropsSI('H','T',(11.67+273.15),'P',101325,'R',1); #corresponding minimum enthalpy
    print h_oa_min
    Coefs= getDesiccantCoeffs(TrainingData,h_oa_min) # regression is a funciton of MODEL variables, rather than AHU system variables, since the AHU system is not modeled.  specific regression variables are outdoor air enthalpy, outdoor air enthalpy squared, hot water inlet temperature, and outdoor air CFM (which may be a constant value when the fan is on)
    
    if T_HW_out_available =='yes'or T_HW_out_available =='Yes'or T_HW_out_available =='YES':
        Coefs2= getTrainingMFR(TrainingData) # if hot water outlet temperature from regeneration coil is available, use historical data on hot water coil inlet and outlet temperatures along with valve command to build regression.

    
h_oa_curr = HAPropsSI('H','T',(T_OA+273.15),'P',101325,'R',RH_OA); # outdoor air enthalpy in current timestep
w_oa_curr = HAPropsSI('W','T',(T_OA+273.15),'P',101325,'R',RH_OA); # outdoor air humidity ratio in current timestep

if T_HW_out_available =='yes'or T_HW_out_available =='Yes'or T_HW_out_available =='YES':
    deltaT_RegenCoil= Coefs2[0]*h_oa_curr+Coefs2[1] # if hot water outlet temperature from regeneration coil is available, get estimate of delta T as a funciton of outdoor air enthalpy.

COP_thermal= 0.2482+4.171529*w_oa_curr-0.00019*T_OA+0.000115*pow(T_OA,2) # Thermal COP is the reduction in cooling load divided by the heat input to the desiccant wheel. The regression equation here is based on Figure 6 in http://www.nrel.gov/docs/fy05osti/36974.pdf

deltaEnthalpy= Coefs[4]+T_HW*Coefs[3]+h_oa_curr*Coefs[2]+pow(h_oa_curr,2)*Coefs[1]+cfm_OA*Coefs[0] #Calculate the change in enthalpy from the outdoor air to the conditioned air stream, prior to mixing box (or cooling coil if it is a 100% OA unit)
if w_oa_curr < w_oa_min or Fan_Status==0: # No useful enthalpy reduction if outdoor air humidity ratio is already below saturated conditions at the cooling coil outlet temperature used for dehumidification
    deltaEnthalpy=0
elif deltaEnthalpy < (min_h-h_oa_curr): # Useful enthalpy reduction should be limited by enthalpy of the supply air at saturation for the cooling coil in dehumidification mode
    deltaEnthalpy=min_h-h_oa_curr


rho_OA=0.0000175*pow(T_OA,2)-(0.00487*T_OA)+1.293 # density of outdoor air in kg/m3 as a function of outdoor air temperature in degrees C
MFR_OA= cfm_OA*rho_OA/pow(3.28,3)/60  # calculate mass flow rate of outdoor air, given volumetric flow rate and air density
Q_Cool_Out=-deltaEnthalpy*MFR_OA  #output: reduction in cooling load (W)
Q_HW_In=Q_Cool_Out/COP_thermal #output: hot water loop heat input (W)
MFR_HW_In = Q_HW_In/cp_water/deltaT_RegenCoil/1000 #output: mass flow rate of water in heating coil [kg/s]
T_HW_Out= T_HW-deltaT_RegenCoil*Fan_Status #output: hot water temperature outlet from regeneration coil [C]
Q_Elec_In= WheelPower #output: Electricity Input [W]
if Q_Cool_Out==0:
    Q_Elec_In= 0

print Q_Cool_Out
print Q_HW_In
print Q_Elec_In
print MFR_HW_In
print T_HW_Out
    

