
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import math

I_L='Normal'
HistPrices= pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\Historical Prices.csv', header=0) #A database of historical prices.  No associated tme of day is required.
PriceForecast= pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\price_forecast.csv', header=0) # Forecast electricity price organized by hours in the future. first row is current hour, second row is next hour, etc.  24 hour forecast required.
ChargeHistory1= pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\chargeHistoryVehicle1.csv', header=0) #Historical charging details for current charge of curent vehicle.  At least timestamp and power are required fields.  State of Charge (SOC) is optional
P_Thresh_curves= pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\P_thresh.csv', header=0) #characteristic price threshold curves as a funciton of SOC.  These curves do not need to change, but can be modified to customize response pattern
F_Thresh_curves= pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\F_thresh.csv', header=0) #characteristic time fraction (charging) curves as a funciton of SOC.  These curves do not need to change, but can be modified to customize response pattern
T_Thresh=0.5 #minimum hours in charging cycle
controlTimestep=5 #minutes between dispatch control of charging stations
minCycle=math.ceil(T_Thresh/(float(controlTimestep)/60)) # minumum number of timesteps in a charging cycle (between periods of interruption)
controlTimestep= 60/controlTimestep # converting to timesteps per hour
ForecastPower=np.zeros((24*controlTimestep), dtype=np.float64) #initializing; Forecast power by timestep
ForecastPowerHourly=np.zeros((24), dtype=np.float64) #initializing; Forecast power by hour
ForecastPrice=PriceForecast['Price Forecast'].values 
P_thresh_Percentile=90 # Nominal price threshold for interruption of charging, as a percentile of historical price data
RatedPower=3 #Initial asusmption for rated power of charging station; overridden based on accumulated historical charging data later.
P_thresh_Default= np.percentile(HistPrices, P_thresh_Percentile) # Nominal price threshold for interruption of charging, in $/kWh
                                

if I_L=='Conservative': #I_L is the "interruption level"
    F_thresh=0.85
elif I_L=='Normal':
    F_thresh=0.80;
else:
    F_thresh=0.75;
    
ChargingEfficiency=0.85 # Assumed efficiency of the charging process.  Charging stations will have to provide more power than is actually stored in the battery 
ChargeDefault=20 #kWh; for Scenario 1, typical charge required; when state of charge is unknown
ChargeDefault=ChargeDefault/ChargingEfficiency
FullBatteryChargeDefault=30 #kWh; typical full battery charge, if not enough data is available to estimate
FullBatteryChargeDefault=FullBatteryChargeDefault/ChargingEfficiency

SOCAvailable='No'
if SOCAvailable=='No': # execute Scenario 1, where state of charge is not measured by the charge station
    Time= ChargeHistory1['Time'].values # Time must be formatted as fractional time of day
    Power= ChargeHistory1['Power'].values # Charging power in kW
    Rows=len(Power)

    #------------------------Search the Charging history and determine the total energy charged during the session, and the length of the current charging cycle (after any interruptions)
    kWhTotal=0  # Inintializing; total energy charged during charging session, kWh
    TimestepsCurrChargeCycle=0 # Inintializing; Number of timesteps since last interrruption or since start of charging session, if no interruptions yet
    for i in range(1,Rows): 
        kWhCurr=24*(Time[i]-Time[i-1])*Power[i-1] # based on time between data entries and power at the start of each interval, energy charged is calculated
        kWhTotal=kWhTotal+kWhCurr 
        if Power[i-1]>0.5:
            TimestepsCurrChargeCycle=TimestepsCurrChargeCycle+1 # count the number of data entries in the current charge cycle after last interruption
        else:
            TimestepsCurrChargeCycle=0 # reset the count if the charge session in interrupted.
            

    ChargeLeft= ChargeDefault-kWhTotal # best guess at remaining charge needed to be supplied by the charge station, given that no SOC data is available
    for i in range(0,24*controlTimetep): # forecast charging patterns over the next 24 hours.  We need to simulate first on a shorter timescale than 1 hour, and later will compile the hourly averages from this data
        
        Timesteps=len(Power) #total number of previous timesteps, including both historical data, and forecasted data up to the current forecasting timestep
        
        #---------------Calculate the fraction of time spent in charging mode and update the (assumed) rated power based on actual measured power from charging stations
        countCharging=0 #initializing; total number of timesteps during which the vehicle was charging
        if Power[Timesteps-1]>0.5: # if the vehicle is charging in current timestep.  Disregard slightly positive values (below 500 W)
            RatedPowerCount=0  #Initialize summation of power
        for j in range (1, Timesteps+1): 
            if Power[j-1]>0.5: 
                countCharging=countCharging+1 # count the timesteps during which the vehicle was charging
                RatedPowerCount=RatedPowerCount+Power[j-1] #sum the delivered power during timesteps in which the vehicle was charging
        fCharging=float(countCharging)/float(Timesteps) # calculated the cumulative fraction of time (including historical and forecast period up til now) that the vehicle has been charging
        if Power[Timesteps-1]>0.5:
            RatedPower=float(RatedPowerCount)/float(countCharging) # average power delivered to the battery during charging 
        
        
        if Power[Rows-1]>0 and i==0:
            ForecastPower[i]=RatedPower  #Start with the assumption that if the vehicle was charging in the previous timestep, it will still be charging in this timestep, regardless of the guess about charge left (which is actually unknown)  
            # NOTE that ForecastPower[0] is the CURRENT DISPATCH COMMAND and should be used to actuate charge station.
        
        #Forecasting beyond current timestep (i>0).  Charge at rated power, unless all criteria for charge interruption are met.
        if ChargeLeft>0 and i>0: 
            if ForecastPrice[math.floor(i/controlTimetep)]>=P_thresh_Default and fCharging >= F_thresh and (TimestepsCurrChargeCycle >= minCycle or TimestepsCurrChargeCycle==0):
                ForecastPower[i]=0
            elif i>0:
                ForecastPower[i]=RatedPower
                
        Power = np.append(Power, ForecastPower[i]) #update the array 'Power' based on the forecast operation in current forecast timestep
        kWhTotal=kWhTotal+ForecastPower[i]/controlTimetep # update total energy delivered from charging station
        
        # update number of timestep in (now forecast) charge cycle based on current forecast timestep
        if ForecastPower[i]>0.5:
            TimestepsCurrChargeCycle=TimestepsCurrChargeCycle+1
        else:
            TimestepsCurrChargeCycle=0
        ChargeLeft= ChargeDefault-kWhTotal #update charge remaining to be delivered after current forecast timestep

    # compile the hourly averages of forecast power over the next 24 hours
    k=0
    for i in range(0,24):
        for j in range (0,controlTimestep-1):
            ForecastPowerHourly[i]=ForecastPowerHourly[i]+ForecastPower[k]/float(controlTimestep)
            k=k+1
            
else: # execute scenario #2, where the SOC is measured by the charging station and sent to Volttron
    Time= ChargeHistory1['Time'].values # Time must be formatted as fractional time of day
    Power= ChargeHistory1['Power'].values #Power from charging history
    SOCHistory=ChargeHistory1['SOC'].values  # SOC's from charging history
    SOCLookupP=P_Thresh_curves['SOC'].values # corresponding SOC's for the P_Thresh curve
    SOCLookupF=F_Thresh_curves['SOC'].values # corresponding SOC's for the F_Thresh curve
    P_thresh = P_Thresh_curves[I_L].values #Find the right P_Thresh curve, depending on the interruption level selected
    F_thresh = F_Thresh_curves[I_L].values# Find the right F_Thresh curve, depending on the interruption level selected
    
    #------------------------Search the Charging history and determine the total energy charged during the session, and the length of the current charging cycle (after any interruptions)
    kWhTotal=0  #Next 12 lines Same as Scenario 1
    Rows=len(Power)
    TimestepsCurrChargeCycle=0
    if Rows>1:
        for i in range(1,Rows): 
            kWhCurr=24*(Time[i]-Time[i-1])*Power[i-1]
            kWhTotal=kWhTotal+kWhCurr
            if Power[i-1]>0.5:
                TimestepsCurrChargeCycle=TimestepsCurrChargeCycle+1
            else:
                TimestepsCurrChargeCycle=0

        dSOC= SOCHistory[Rows-1]-SOCHistory[0] #calculate total change in state of charge over the charging history
        kWhPerCharge=kWhTotal/dSOC # calcualte total kWh needed to fully charge the battery, based on observed response of SOC to charging

    else:
        kWhPerCharge=FullBatteryChargeDefault #if not enough data exisists yet in the charging history file, fall back on default asusmption for full battery charge
    
    ChargeLeft= (1-SOCHistory[Rows-1])*kWhPerCharge #given the most recent SOC, and calculated full battery charge, calculate charge remaining
    
    CurrSOC=SOCHistory[Rows-1] #initialize current SOC for the forecast period
    for i in range(0,24*controlTimetep): # forecast charging patterns over the next 24 hours.
        
        Curr_P_Thresh=np.interp(CurrSOC, SOCLookupP, P_thresh) #interpolate P_Thresh curve, to find cuurent price threshold based on current state of charge
        Curr_F_Thresh=np.interp(CurrSOC, SOCLookupF, F_thresh) #interpolate F_thresh curve, to fined current fraction of time spent charging threshold based on current state of charge
        
         #---------------Calculate the fraction of time spent in charging mode and update the (assumed) rated power based on actual measured power from charging stations
        Timesteps=len(Power) #Next 11 lines same as Scenario 1
        countCharging=0
        if Power[Timesteps-1]>0.5:
            RatedPowerCount=0
        for j in range (1, Timesteps+1):
            if Power[j-1]>0.5:
                countCharging=countCharging+1
                RatedPowerCount=RatedPowerCount+Power[j-1]
        fCharging=float(countCharging)/float(Timesteps)
        if Power[Timesteps-1]>0.5:
            RatedPower=float(RatedPowerCount)/float(countCharging)


        if SOCHistory[Rows-1]<1 and i==0:  #Start with the assumption that if the vehicle is not yet fully charged, it will still be charging in this timestep.
            ForecastPower[i]=RatedPower # Note that ForecastPower[0] is the current dispatch command.
            
        if ChargeLeft>0 and i>0: #Forecasting beyond current timestep (i>0).  Charge at rated power, unless all criteria for charge interruption are met.
            if ForecastPrice[math.floor(i/controlTimetep)]/P_thresh_Default >=Curr_P_Thresh and fCharging >= Curr_F_Thresh and (TimestepsCurrChargeCycle >= minCycle or TimestepsCurrChargeCycle==0):
                ForecastPower[i]=0
            elif i>0:
                ForecastPower[i]=RatedPower
                
        Power = np.append(Power, ForecastPower[i]) #Next 3 lines same as Scenario 1
        kWhTotal=kWhTotal+ForecastPower[i]/controlTimetep
        ChargeLeft= ChargeLeft-ForecastPower[i]/controlTimestep
        CurrSOC=1-ChargeLeft/kWhPerCharge #Update forecast SOC based on forecast charging
        if ForecastPower[i]>0.5: #Same as Scenario 1
            TimestepsCurrChargeCycle=TimestepsCurrChargeCycle+1
        else:
            TimestepsCurrChargeCycle=0

    #compile the hourly averages of forecast power over the next 24 hours. Same as scenario 1
    k=0 
    for i in range(0,24):
        for j in range (0,controlTimestep):
            ForecastPowerHourly[i]=ForecastPowerHourly[i]+ForecastPower[k]/float(controlTimestep)
            k=k+1
print ForecastPowerHourly
            
            
    

