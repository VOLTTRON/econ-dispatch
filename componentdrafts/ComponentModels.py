
# coding: utf-8

# ### Function - Inverter: Output:Electricity

# In[146]:

def InverterElecOut(PLF,Efficiency, ElecIn,Cap):
    Eff= np.interp(ElecIn/Cap, PLF, Efficiency)
    return Eff*ElecIn                                                    


# In[ ]:




# # Function - HeatRecovery: Output:WaterTemperature

# In[147]:

def HeatRecoveryWaterOut(PrimeMoverFluid,HotWaterFluid,Method,T_water_i,T_ex_i,MFR_water,MFR_ex,CurrSpeed,TrainingData,HeatExchagerArea,DefaultEffectiveness):
    import math
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
        
    C_ex = MFR_ex*cp_ex
    C_water = MFR_water*cp_water
    C_min = min(C_ex,C_water)
    C_max = max(C_ex,C_water)
    C_r = C_min/C_max
    
    if Method ==1:
        U=650 #w/m2-K
        UA=U*HeatExchagerArea #W-K
        NTU=UA/C_min
        Eff=(1-math.exp(-1*NTU*(1-C_r)))/(1-C_r*math.exp(-1*NTU*(1-C_r))) #relation for concentric tube in counterflow
        T_water_o=T_water_i+Eff*C_min/C_water*(T_ex_i-T_water_i)
    elif Method==2:
        Eff=DefaultEffectiveness
        T_water_o=T_water_i+Eff*C_min/C_water*(T_ex_i-T_water_i)
    elif Method==3:
        T_water_o = C[0]+T_water_i+C[1]*CurrSpeed/MFR_water*(T_ex_i-T_water_i)
    
    return T_water_o
    
    


    


# # Function - HeatRecovery: GetHeatRecovered

# In[148]:

def GetHeatRecovered(HotWaterLoopFluid,T_water_i,TWaterOut,MFR_water):   
    if HotWaterLoopFluid == 'Water':
        cp_water=4186 #j/kg-K    
    elif HotWaterLoopFluid == 'GlycolWater30':
        cp_water=3913 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater50':
        cp_water=3558 #j/kg-K
    else:
        cp_water=4186 #j/kg-K
    HeatRecovered = MFR_water*cp_water*(TWaterOut-T_water_i)/1000 #kW
    return HeatRecovered


# # Function - HeatRecovery: GetRegressionHeatRecovery

# In[149]:

def GetRegressionHeatRecovery(TrainingData):
    from scipy import stats
    import numpy as np
    
    Twi= TrainingData['T_wi'].values
    Two= TrainingData['T_wo'].values
    Texi= TrainingData['T_ex_i'].values
    Speed=TrainingData['Speed'].values
    MFRw=TrainingData['MFR_water'].values
    Proportional= Speed/MFRw*(Texi-Twi)
    Output=Two-Twi
    
    x = Proportional
    y = Output
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    C = [intercept, slope]
    return C 


# In[ ]:




# # Function - ThermalStorage: GetNodeTemperatures

# In[150]:

def getNodeTemperatures(dt, MFR_chW, T_chW, MFR_abW, T_abW, MFR_chwBldg,T_chwBldgReturn,Nodes_tminus1,TankVolume,TankHeight,Fluid,InsulatedTank):
    
    import numpy.matlib
    
    T_zone =21 #assumed constant teperature of thermal zone
    h=4.00 #assumed convection heat transfer coefficient
    n = len(Nodes_tminus1)
    if InsulatedTank == 'yes' or InsulatedTank =='Yes':
        k_tank=0.03 #thermal conductivity of tank wall and insulation, in W/m-K
        x_tank=0.0254 #thickness of tank wall and insulation, in m
    else:
        k_tank=16.0 #thermal conductivity of tank wall, in W/m-K
        x_tank=0.01 #thickness of tank wall, in m   
        
    if Fluid == 'Water' or Fluid == 'water':
        cp_fluid=4186 #j/kg-K 
        rho_fluid=1.000 #kg/L
        k_fluid= 0.609 # W/m-K
    elif Fluid == 'GlycolWater30' or Fluid == 'glycolwater30':
        cp_fluid=3913 #j/kg-K
        rho_fluid=1.026 #kg/L
        k_fluid = 0.4707 # W/m-K
    elif Fluid == 'GlycolWater50' or Fluid == 'glycolwater50':
        cp_fluid=3558 #j/kg-K
        rho_fluid=1.041 #kg/L
        k_fluid = 0.378 # W/m-K
    else:
        cp_fluid=4186 #j/kg-K
        rho_fluid=1.000 #kg/L
        k_fluid= 0.609 # W/m-K
        
        

        
    m_fluid = rho_fluid*TankVolume/n
    
    
    pi=3.141596
    TankRadius = pow(((TankVolume/1000)/(pi*TankHeight)),0.5)
    Circ = pi*2*TankRadius
    ExtAreaPerSegment = Circ*TankHeight/n
    CrossSecArea = pi*TankRadius*TankRadius
    MFR_netDown=MFR_chwBldg-MFR_abW-MFR_chW
    MFR_inmix = 0.50*TankVolume*rho_fluid/(n*dt) #inversion mixing flow rate, set arbitrarily at 50% of flow rate for full mixing of one node to the next in one timestep

    i=0
    Nodes_t=[]
    for i in range(0,len(Nodes_tminus1)):
        
        # Calculate combined conduction and convection from tank vertical walls (standby losses)
        L= TankHeight #Characteristic Length for convection
        
        if InsulatedTank == 'yes' or InsulatedTank =='Yes':
            BulkAir=(T_zone)
        else:
            BulkAir=(Nodes_tminus1[i]+T_zone)/2
        rho_air=0.0000175*pow(BulkAir,2)-(0.00487*BulkAir)+1.293
        mu= ((0.0049*BulkAir)+1.7106)*pow(10,-5)
        Cp_air=1004.0
        k_air=0.00007*BulkAir+0.0243 # Thermal conductivity of air, in W/m-K
        g=9.81
        beta = 1/(BulkAir+273.15)
        mu_v=mu/rho_air
        alpha_air = k_air/(rho_air*Cp_air)
        Pr = Cp_air*mu/k_air # dimensionless Prandtl Number; Approximately 0.7 to 0.8 for air and other gases
        Ra=g*beta*pow(pow(Nodes_tminus1[i]-T_zone,2),0.5)*pow(L,3)/(mu_v*alpha_air)
        
        # Performing Nusselt number calculation in steps because of problems with a single calculation
        Nu1=0.387*pow(Ra,0.16666)
        Nu2= pow((0.492/Pr),0.5625)
        Nu3= pow(Nu2+1,0.296296)
        Nu4=pow((0.825+Nu1/Nu3),2) # Nu4 is non-dimensional Nusselt number for a vertical plate from Equation 9.27 in Fundamental of Heat and Mass Transfer, 6th Edition
    
        h_v=Nu4*k_air/L
   

        q_conv_v= (T_zone-Nodes_tminus1[i])/((1/h_v)+(x_tank/k_tank))
        Q_conv_v=ExtAreaPerSegment*q_conv_v
        
        #Calculate convection from the top of the tank (Q_conv_t)
        if i==0:
            L= 2*TankRadius
            Ra=g*beta*pow(pow(Nodes_tminus1[i]-T_zone,2),0.5)*pow(L,3)/(mu_v*alpha_air)
            Nu=0.27*pow(Ra,0.25)
            h_h_t=Nu*k_air/L
            q_conv_t= (T_zone-Nodes_tminus1[i])/((1/h_h_t)+(x_tank/k_tank))
        else:
            q_conv_t=0
        Q_conv_t=CrossSecArea*q_conv_t
        
        #Calculate convection from the bottom of the tank (Q_conv_b)
        if i==(len(Nodes_tminus1)-1):
            L= 2*TankRadius
            Ra=g*beta*pow(pow(Nodes_tminus1[i]-T_zone,2),0.5)*pow(L,3)/(mu_v*alpha_air)
            Nu=0.15*pow(Ra,0.3333)
            h_h_b=Nu*k_air/L
            q_conv_b= (T_zone-Nodes_tminus1[i])/((1/h_h_b)+(x_tank/k_tank))
        else:
            q_conv_b=0
        Q_conv_b=CrossSecArea*q_conv_b     
        
        # Calculate conduction between current node and node above
        if i>0:
            q_cond_nminus1=  (Nodes_tminus1[i-1]-Nodes_tminus1[i])*k_fluid/(TankHeight/n)
            Q_cond_nminus1=  CrossSecArea*q_cond_nminus1
        else:
            Q_cond_nminus1=0
            
        # Calculate conduction between current node and node below
        if i<(len(Nodes_tminus1)-1):
            q_cond_nplus1=  (Nodes_tminus1[i+1]-Nodes_tminus1[i])*k_fluid/(TankHeight/n)
            Q_cond_nplus1=  CrossSecArea*q_cond_nplus1
        else:
            Q_cond_nplus1=0
        
        # Calculate heat from water pushing down from above
        if MFR_netDown>0:
            if i>0:
                Q_flow_down= MFR_netDown*cp_fluid*(Nodes_tminus1[i-1]-Nodes_tminus1[i])
            else:
                Q_flow_down=0
        else: 
            Q_flow_down=0
        
        # Calculate heat from water pushing up from below
        if MFR_netDown<=0:
            if i<(len(Nodes_tminus1)-1):
                Q_flow_up= (0-MFR_netDown)*cp_fluid*(Nodes_tminus1[i+1]-Nodes_tminus1[i])
            else:
                Q_flow_up=0
        else: 
            Q_flow_up=0
        
        # Calculate cooling at bottom node from chiller and absorption chiller
        if (MFR_chW+MFR_abW>0) and i==len(Nodes_tminus1)-1:
            T_Source = (T_chW*MFR_chW+T_abW*MFR_abW)/(MFR_chW+MFR_abW)
            Q_Source = (MFR_chW+MFR_abW)*cp_fluid*(T_Source-Nodes_tminus1[i])
        else:
            Q_Source =0
        
        # Calculate heating at top from return building water
        if (MFR_chwBldg>0) and i==0:
            Q_Use = (MFR_chwBldg)*cp_fluid*(T_chwBldgReturn-Nodes_tminus1[i])
        else:
            Q_Use =0
        
        #Calculate inversion mixing from above
        if i>0:
            if Nodes_tminus1[i-1]<Nodes_tminus1[i]:
                Q_inmix_above= MFR_inmix*cp_fluid*(Nodes_tminus1[i-1]-Nodes_tminus1[i])
            else:
                Q_inmix_above=0
        else:
            Q_inmix_above=0
   
        #Calculate inversion mixing from below
        if i<(len(Nodes_tminus1)-1):
            if Nodes_tminus1[i+1]>Nodes_tminus1[i]:
                Q_inmix_below= MFR_inmix*cp_fluid*(Nodes_tminus1[i+1]-Nodes_tminus1[i])
            else:
                Q_inmix_below=0
        else:
            Q_inmix_below=0
        
        Q_total=Q_cond_nminus1+Q_cond_nplus1+Q_conv_v+Q_flow_down+Q_flow_up+Q_Source+Q_Use+Q_inmix_above+Q_inmix_below+Q_conv_b+Q_conv_t

        Nodes_t.append(Nodes_tminus1[i]+Q_total*dt/(m_fluid*cp_fluid))     
    
    return Nodes_t


# # Function - Battery:GetChargingParameters

# In[151]:

def GetChargingParameters(TrainingData, Capacity):
    from scipy import stats
    import numpy as np
    
    Time= TrainingData['Time'].values
    Current= TrainingData['I'].values
    PowerIn= TrainingData['Po'].values
    SOC=TrainingData['SOC'].values
    x=len(Time)
    PrevTime = []
    CurrTime = []
    CurrPower= []
    PrevPower= []
    CurrSOC=[]
    PrevSOC=[]
    CurrentI=[]
    ChargeEff=[]
    Slope_RI=[]
    for i in range(0,x+1):
        if i>1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrPower.append(PowerIn[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
            CurrentI.append(Current[i-1])
    for i in range(0,x-1):
        ChargeEff.append((CurrSOC[i]-PrevSOC[i])*Capacity/((CurrTime[i]-PrevTime[i])*24)/CurrPower[i])
        Slope_RI.append(0-CurrentI[i]*CurrentI[i]/CurrPower[i])
        
    x = Slope_RI
    y = ChargeEff
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    C = [intercept, slope]
    return C


# # Function - Battery:GetDisChargingParameters

# In[152]:

def GetDisChargingParameters(TrainingData, Capacity):
    from scipy import stats
    import numpy as np
    
    Time= TrainingData['Time'].values
    Current= TrainingData['I'].values
    PowerIn= TrainingData['Po'].values
    SOC=TrainingData['SOC'].values
    Rows=len(Time)
    PrevTime = []
    CurrTime = []
    CurrPower= []
    PrevPower= []
    CurrSOC=[]
    PrevSOC=[]
    CurrentI=[]
    Y=[]
    X=[]
    for i in range(0,Rows+1):
        if i>1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrPower.append(PowerIn[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
            CurrentI.append(Current[i-1])
    for i in range(0,Rows-1):
        Y.append((CurrSOC[i]-PrevSOC[i])*Capacity/((CurrTime[i]-PrevTime[i])*24)/(CurrentI[i]*CurrentI[i]))
        X.append(CurrPower[i]/(CurrentI[i]*CurrentI[i]))
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    IR_discharge=(0-intercept)
    InvEFFDischarge=1/slope
    C = [IR_discharge, InvEFFDischarge]
    return C


# # Function - Battery:GetIdleParameters

# In[153]:

def GetIdleParameters(TrainingData, Capacity):
    from scipy import stats
    import numpy as np
    
    Time= TrainingData['Time'].values
    SOC=TrainingData['SOC'].values
    Rows=len(Time)
    PrevTime = []
    CurrTime = []
    CurrSOC=[]
    PrevSOC=[]
    Y=[]
    X=[]
    for i in range(0,Rows+1):
        if i>1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
    for i in range(0,Rows-1):
        Y.append((CurrSOC[i]-PrevSOC[i])/((CurrTime[i]-PrevTime[i])*24))
        X.append(CurrSOC[i])
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    C = [slope, intercept]
    return C


# # Function- Battery:GetUpdatedSOC

# In[154]:

def getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevSOC,InvEFF_Charge,IR_Charge,IR_DisCharge,InvEFF_DisCharge,Idle_A,Idle_B):
    import numpy as np
    
    deltaSOC=0 #change in state of charge during this timestep. Initialization of variable
    if InputPowerRequest>0: #for batttery charging
        P2= InputPowerRequest*InvEFF_Charge-InputCurrent*InputCurrent*IR_Charge #P2 is power recieved into battery storage
        deltaSOC= P2*timestep/3600/Capacity 
    if InputPowerRequest<0:
        P2 = InputPowerRequest/InvEFF_DisCharge-InputCurrent*InputCurrent*IR_DisCharge #P2 is power released from battery storage
        deltaSOC= P2*timestep/3600/Capacity
    deltaSOC_self=(Idle_A+Idle_B*PrevSOC)*timestep/3600
    SOC = PrevSOC+deltaSOC+deltaSOC_self
    
    InputPower=InputPowerRequest
    if SOC < minimumSOC:
        if PrevSOC<minimumSOC and InputPowerRequest<0:
            InputPower=0
            SOC = PrevSOC+deltaSOC_self
        if PrevSOC>minimumSOC and InputPowerRequest<0:
            InputPower=InputPowerRequest*(PrevSOC-minimumSOC)/(PrevSOC-SOC)
            InputCurrent=InputCurrent*InputPower/InputPowerRequest
            P2 = InputPower/InvEFF_DisCharge-InputCurrent*InputCurrent*IR_DisCharge
            deltaSOC= P2*timestep/3600/Capacity
            SOC = PrevSOC+deltaSOC+deltaSOC_self
    if SOC > 1:
            InputPower=InputPowerRequest*(1-PrevSOC)/(SOC-PrevSOC)
            InputCurrent=InputCurrent*InputPower/InputPowerRequest
            P2= InputPower*InvEFF_Charge-InputCurrent*InputCurrent*IR_Charge
            deltaSOC= P2*timestep/3600/Capacity
            SOC = PrevSOC+deltaSOC+deltaSOC_self
    if SOC < 0:
            SOC=0
        
    
    Outputs = [SOC, InputPower]
    return Outputs


# # Main Program - Inverter

# In[155]:

get_ipython().magic(u'matplotlib inline')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import * 
from IPython.display import Javascript, display
from collections import OrderedDict



InverterCurve = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\InverterCurve.csv', header=0)
InverterCurveArray=pd.DataFrame(InverterCurve)
PLFvals= InverterCurve['PLF'].values
EffVals=InverterCurve['Efficiency'].values
ElecIn=10
InverterCapacity=150.0
ElecOut=InverterElecOut(PLFvals,EffVals,ElecIn,InverterCapacity)
print ElecOut


# # Main Program - Heat Recovery

# In[156]:

# Inputs - scalar values for now, representing current readings
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


# # Main Program - Chilled Water Thermal Storage

# In[157]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib

TankVolume = 37800.0 # Tank Volume in L
TankHeight = 2.0 # Tank Height in m
InsulatedTank ='No'
Fluid = 'water'
timestep = 60
MFR_chW = 50.4 # Flow rate of chilled water to the tank from the chiller, in kg/s
T_chW = 7 # Temperature of chilled water from the chiller, in degrees C
MFR_abW = 10 # Flow rate of chilled water to the tank from the absorption chiller, in kg/s
T_abW = 10 # Temperature of chilled water from the absorption chiller, in degrees C
MFR_chwBldg = 30 # Flow rate of chilled water to the building loads from the tank
T_chwBldgReturn = 14 # return chilled water temperature from bulding in degress C

Nodes = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6] # initialization of temperatures in tank


for i in range(1,2):
    Nodes = getNodeTemperatures (timestep, MFR_chW, T_chW, MFR_abW, T_abW, MFR_chwBldg,T_chwBldgReturn,Nodes,TankVolume,TankHeight,Fluid,InsulatedTank)
    print Nodes


# # Main Program - Battery

# In[158]:

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
    



# # Main Program: Desiccant Wheel
# 

# In[159]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib
import CoolProp
from CoolProp.HumidAirProp import HAPropsSI


WheelPower = 300 #Electric power in Watts consumed to run desiccant system, typically including desiccant wheel motor and heat recovery wheel motor
cfm_OA=22000 # CFM of outdoor air 
GetTraining = 'No' 
T_OA = 35 # outdoor air temperature, from building sensor, in degrees C
RH_OA = 0.65 # outdoor air relative humidity, from building sensor in kg/m3
T_HW = 80 # Hot water loop temperature in degrees C
Regen_Valve = 0.4 # Hot water valve position on regeneration hot water coil (fraction)
COP_cool=3.5  # Assumed COP of alternative cooling source feeding downstream cooling coil (typically chiller or DX coil)
deltaEnthalpy = -10000

if GetTraining == 'yes'or GetTraining =='Yes'or GetTraining =='YES':
    TrainingData = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\BatteryCharging.csv', header=0) #this should
    #reference a data file from recent battery charg


h = HAPropsSI('H','T',(T_OA+273.15),'P',101325,'R',RH_OA); 
print h



rho_OA=0.0000175*pow(T_OA,2)-(0.00487*T_OA)+1.293 # density of outdoor air in kg/m3 as a function of outdoor air temperature in degrees C
MFR_OA= cfm_OA*rho_OA/pow(3.28,3)/60  # calculate mass flow rate of outdoor air, given volumetric flow rate and air density
ElecSavingsDesiccant=-deltaEnthalpy*MFR_OA/COP_cool

print ElecSavingsDesiccant
print rho_OA
print MFR_OA


# # GetDesiccantCoeffs

# In[ ]:

def getUpdatedSOC(Capacity, minimumSOC, InputPowerRequest, InputCurrent, timestep, PrevS
import numpy as np
    

return Outputs

