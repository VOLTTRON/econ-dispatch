import math

import numpy as np
import pandas as pd


# This class is just here so we can maintain the structure of the original code for translation
# Checking to see if a value exists in a structure as in FuelCell_Calibrate is not the Right Thing
class Coefficient(object):
    def __getattr__(self, name):
        return self.__dict__.get(name, None)

    def __str__(self):
        return 'Coef\n' + '  \n'.join([str(x) for x in sorted(self.__dict__.items())])


def first_five_percent_or_50(array):
    nonzero = np.nonzero(array)[0]
    n_nonzero = len(nonzero)

    five_percent = int(math.ceil(n_nonzero / 20))
    n_take = max(50, five_percent)

    return nonzero[:n_take]



def FuelCell_Calibrate(Type, Fuel, NetPower, FuelFlow, Time, InletTemperature, AirFlow, ExhaustTemperature, Voltage, Current, AncillaryPower, Cells, CellArea, StackDeltaT, Coef):
    ## Inputs
    #Type: string of what kind of FC 'PEM', 'PAFC', 'MCFC', or 'SOFC' (NOT optional)
    #Fuel: string of what type of fuel is provided 'CH4' or 'H2' (NOT optional)
    #NetPower: power in kW (NOT optional)
    #FuelFlow: in kg / s  (NOT optional)
    #Time: cumulative hours that the FC has been running (NOT optional)
    #InletTemperature (ambient conditions): in Celsius (NOT optional)
    #AirFlow: in kg / s
    #ExhaustTemperature: in Celsius
    #Voltage: Stack voltage in V
    #Current: Cumulative current (if there are multiple stacks) in Amps
    #AncillaryPower: Power internally consumed by blowers pumps and fans (kW)
    #CellTemperature: nominal operating temperature in Celsius
    #Cells: number of cells in the stack
    #CellArea: effective cell area in cm^2: Power (kW) = current density (A / cm^2) * Area (cm^2) * Voltage (V) / 1000
    #StackDeltaT: nominal temperature difference accross the stack in degrees Celsius
    ## Outputs:
    #Coef: structure of previously or user defined coefficients, default isempty []

    ##---###
    ## Calculations
    #Fueltype

    Coef = Coefficient()

    if not Coef.Fuel:
        if Fuel:
            Coef.Fuel = Fuel
        else:
            Coef.Fuel = 'CH4'

    if Coef.Fuel == 'CH4':
        n = 8.0 # # of electrons per molecule (assuming conversion to H2)
        LHV = 50144.0 #Lower heating value of CH4 in kJ / kg
        m_fuel = 16.0#molar mass kg / kmol
    elif Coef.Fuel == 'H2':
        n = 2.0 # # of electrons per molecule (assuming conversion to H2)
        LHV = 120210.0 #Lower heating value of H2 in kJ / kg
        m_fuel = 2.0#molar mass kg / kmol

    Coef.kg_per_Amp = m_fuel / (n * 1000 * 96485)
    Coef.LHV = LHV
    E1 = 1000 * Coef.LHV * Coef.kg_per_Amp#voltage potential of fuel (efficiency = utilization * voltage / E1)

    #Power
    NetPower[NetPower<0] = 0
    sortPow = np.sort(NetPower) #NetPower
    if not Coef.NominalPower:
        Coef.NominalPower = round(sortPow[int(round(len(sortPow) * 0.98))]) #assume 2# outliers in data

    nPower = NetPower / Coef.NominalPower
    Efficiency = NetPower / (FuelFlow * LHV)
    Efficiency[np.isnan(Efficiency)] = 0
    Efficiency[np.isinf(Efficiency)] = 0

    valid = (Efficiency > 0.01) & (Efficiency < 0.7)
    a = np.polyfit(NetPower[valid], Efficiency[valid], 2)

    valid = (Efficiency > 0.8 * (a[0] * NetPower ** 2 + a[1] * NetPower + a[2])) & (Efficiency < 1.2 * (a[0] * NetPower ** 2 + a[1] * NetPower + a[2]))
    nominal = (NetPower > Coef.NominalPower * 0.95) & (NetPower < Coef.NominalPower * 1.05) & valid

    #stack temperature gradient
    if not Coef.StackDeltaT:
        if StackDeltaT:
            Coef.StackDeltaT = StackDeltaT
        else:
            Coef.StackDeltaT = 100

    ## of cells
    if not Coef.Cells:
        if Cells:
            Coef.Cells = Cells
        elif Voltage:
            util = 0.8 - (0.6-mean(Efficiency(nominal)))#estimate fuel utilization at peak efficiency
            Coef.Cells = round(mean(util * Voltage(nominal) / (Efficiency(nominal) * E1))) #estimate of the # of cells: Eff = util * V / Cells / E1
        elif Current:
            Utilization = Efficiency(nominal)^ 0.5 #assumed fuel utilization
            Coef.Cells = round(mean((FuelFlow(nominal) / m_fuel) * Utilization * n * 1000 * 96485. / Current(nominal)))
        else:
            Coef.Cells = round(Coef.NominalPower)#Assumption 1kW cells

    ##At least 2 of 3 must be known (Current, Voltage, & Ancillary Power), or some assumptions will be made:
    if Voltage and Current:
        # if voltage & current are known, ancillary power is easily computed.
        TotalPower = Voltage * Current / 1000 #stack Power in kW
        AncillaryPower = TotalPower - NetPower #first guess of ancillary power for actual operation
    elif not AncillaryPower:
        #estimate an air flow
        if not AirFlow:
            ExhaustFlow = (FuelFlow * LHV - NetPower) / (1.144 * 2 * Coef.StackDeltaT) #flow rate in kg / s with a specific heat of 1.144kJ / kg * K
        else:
            ExhaustFlow = AirFlow

        nominalEfficiency = np.mean(Efficiency[nominal])
        a = np.polyfit(InletTemperature[nominal]-18, FuelFlow[nominal] * LHV * nominalEfficiency-Coef.NominalPower, 1)
        Coef.AncillaryPower = np.zeros(3)
        Coef.AncillaryPower[2] = 0.5 * a[0] / np.mean(ExhaustFlow[nominal])
        Coef.AncillaryPower[1] = (0.05 * Coef.NominalPower - 0.5 * a[0] * np.mean(InletTemperature[nominal] - 18)) / np.mean(ExhaustFlow[nominal] + FuelFlow[nominal])#makes it so nominal ancillary power is 15%
        AncillaryPower = Coef.AncillaryPower[0] * FuelFlow ** 2  + Coef.AncillaryPower[1] * FuelFlow + Coef.AncillaryPower[0] * ExhaustFlow ** 2 + Coef.AncillaryPower[1] * ExhaustFlow + Coef.AncillaryPower[2] * (InletTemperature-18) * ExhaustFlow

    TotalPower = AncillaryPower + NetPower
    if Voltage:
        #if we know voltage & ancillary power we can find current
        Current = TotalPower * 1000. / Voltage
    elif Current:
        #if we know current& ancillary power we can find voltage
        Voltage = TotalPower * 1000. / Current
    else:
        #otherwise cell area, OCV & ASR are assumed
        #efficiency = (current * voltage - ancillary power) / energy in
        #Voltage = OCV - ASR * I / area
        #these two expressions reduce to a single quadratic of I that must be solved
        if not Coef.NominalASR:
            if Type == 'SOFC':
                Coef.NominalASR = 0.25
            elif Type == 'PAFC':
                Coef.NominalASR = 0.5
            elif Type == 'MCFC':
                Coef.NominalASR = 0.75
            elif Type == 'PEM':
                Coef.NominalASR = 0.2
            else:
                Coef.NominalASR = 0.25

        if not Coef.NominalOCV:
            if Type == 'SOFC':
                Coef.NominalOCV = 1
            elif Type == 'PAFC':
                Coef.NominalOCV = 0.8
            elif Type == 'MCFC':
                Coef.NominalOCV = 0.85
            elif Type == 'PEM':
                Coef.NominalOCV = 0.75
            else:
                Coef.NominalOCV = 0.9

        # nData = find(nominal, max(50, ceil(nnz(nominal) / 20)))#1st 5# of data above 75% power, or 50 data points
        nominal5perc = first_five_percent_or_50(nominal)

        # nominal5perc = nominal & ((1:length(nominal))'< nData(end))
        #given OCV & ASR, can find nominal utilization, voltage & current resulting in correct efficiency
        nominalCurrent = np.mean(TotalPower[nominal5perc]) * 1000.0 / (0.8 * Coef.NominalOCV * Coef.Cells)
        if not Coef.Area:
            if not CellArea:
                if Type == 'SOFC':
                    Coef.Area = nominalCurrent / 0.5 #assume a current density of 0.5A / cm^2
                elif Type == 'PAFC':
                    Coef.Area = nominalCurrent / 0.33 #assume a current density of 0.25A / cm^2
                elif Type == 'MCFC':
                    Coef.Area = nominalCurrent / 0.25 #assume a current density of 0.2A / cm^2
                elif Type == 'PEM':
                    Coef.Area = nominalCurrent / 0.5 #assume a current density of 0.5A / cm^2
                else:
                    Coef.Area = nominalCurrent / 0.5 #assume a current density of 0.5A / cm^2
            else:
                Coef.Area = CellArea

        for i in range(4): #find the V & I assuming this OCV and ASR
            nominalVoltage = Coef.Cells * (Coef.NominalOCV - nominalCurrent * Coef.NominalASR / Coef.Area)
            nominalCurrent = 1.2 * np.mean(TotalPower[nominal5perc]) * 1000.0 / nominalVoltage - 0.2 * nominalCurrent

        nominalUtil = nominalCurrent * Coef.Cells / (np.mean(FuelFlow[nominal5perc]) / m_fuel) / (n * 1000 * 96485)

        valid2 = first_five_percent_or_50(NetPower)

        ##Total power = V * I = (OCV-ASR / area * I) * I:  find I & V
        Current = TotalPower[valid2] * 1000 / nominalVoltage
        for i in range(4): #find the V & I assuming this OCV and ASR
            Voltage = Coef.Cells * (Coef.NominalOCV - Current * Coef.NominalASR / Coef.Area)
            Current = 1.2 * TotalPower[valid2] * 1000.0 / Voltage - 0.2 * Current

        Utilization = Current * Coef.Cells / (n * 1000 * 96485) / (FuelFlow[valid2] / m_fuel)
        _pow = Voltage * Current / 1000
        for i in range(len(Utilization)):# fit the maximum utilizations at each power level (steady-state)
            _range = (_pow > 0.99 * _pow[i]) & (_pow < 1.01 * _pow[i])
            if Utilization[i] < max(Utilization[_range]):
                Utilization[i] = 0

        # estimate the controller part-load Utilization vs. Power initially when ASR = nominal ASR
        npow = nPower[valid2]
        npow = npow[Utilization>0]
        Utilization = Utilization[Utilization>0]
        C = np.column_stack(((1-npow) ** 2, (1-npow), np.ones(len(npow))))
        d = Utilization
        Aeq = [0, 0, 1]
        beq = nominalUtil
        A = [[1, 0, 0], [0, 1, 0], [-1, 1, 0]]
        b = [0,0,0]
        # Coef.Utilization = lsqlin(C, d, A, b, Aeq, beq)

        ################################################################################
        #hardcoded until we can get the solution from a function, gen1 Utilization
        Coef.Utilization = [-0.252401442379831, -0.252401442379831, 0.624704478460831]
        ################################################################################

        # recalculate V & I as ASR degrades so that efficiency matches
        # current = reverse function of Utilization: , Power & fuel flow are inputs
        Utilization = Coef.Utilization[0] * (1-nPower)**2 + Coef.Utilization[1] * (1-nPower) + Coef.Utilization[2] #decrease in utilization at part load
        Current = Utilization * FuelFlow / m_fuel * (n * 1000 * 96485) / Coef.Cells

        #voltage is whatever is necessary to produce correct total power
        # degradding ASR is this evolving relationshp between current and voltage
        Voltage = TotalPower * 1000. / Current
        Voltage[np.isinf(Voltage)] = 0
        Voltage[np.isnan(Voltage)] = 0

    # This IF is never entered with the provided test data. Why is it here?
    # Part-Load Utilization is calculated from fuel flow and current.
    if not Coef.Utilization:
        Utilization = Current * Coef.Cells / (n * 1000 * 96485) / (FuelFlow / m_fuel)
        valid2 = (NetPower>Coef.NominalPower * 0.15) & (NetPower<Coef.NominalPower * 0.9) & valid
        nData = find(valid2, max(50, ceil(nnz(valid2) / 5)))#1st 5# of data above 75% power, or 50 data points
        valid2 = valid2 & (range(len(valid2)) < nData[-1])
        # estimate the controller part-load Utilization vs. Power initially when ASR = nominal ASR
        C = [(1-nPower(valid2))**2, (1-nPower(valid2)), ones(nnz(valid2), 1)]
        d = Utilization(valid2)
        A = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
        b = [0, 0, -0.9 * util]
        Coef.Utilization = lsqlin(C, d, A, b)

    #Heat Loss
    if not Coef.StackHeatLoss:
        if not AirFlow:
            Coef.StackHeatLoss = 0.1 #assume 10% heat loss
        else:
            Qgen = Coef.Cells * (1.2532 - Voltage / Coef.Cells) * Current / 1000
            Coef.StackHeatLoss = mean(Qgen - AirFlow * 1.144 * Coef.StackDeltaT) / Coef.NominalPower ##flow rate in kg / s with a specific heat of 1.144kJ / kg * K

    #Air flow from the heat gen - heat loss / deltaT,
    #flow rate in kg / s with a specific heat of 1.144kJ / kg * K
    if not AirFlow:
        AirFlow = (Coef.Cells * Current * (1.2532 - Voltage / Coef.Cells) / 1000 - Coef.StackHeatLoss * NetPower) / (1.144 * Coef.StackDeltaT)

    # adjust errors in energy balance calculations to measured exhaust temperature
    if not Coef.ExhaustTemperature:#deviation from calculated exhaust temp
        if ExhaustTemperature:
            if not InletTemperature:
                InletTemperature = 300
            CalculatedExhaustTemperature = (FuelFlow * LHV - NetPower - Coef.StackHeatLoss * Coef.NominalPower) / (1.144 * AirFlow) + InletTemperature
            Coef.ExhaustTemperature = polyfit(nPower, ExhaustTemperature - CalculatedExhaustTemperature, 2) #adjustment to calculated exhaust temperature
        else:
            Coef.ExhaustTemperature = [0, 0, 0]

    #AncillaryPower
    if Coef.AncillaryPower is None:
        C = [(FuelFlow(nominal)+AirFlow(nominal))^2, (FuelFlow(nominal)+AirFlow(nominal)), (InletTemperature(nominal)-18)]
        d = AncillaryPower
        A = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        b = [0, 0, 0]
        Coef.AncillaryPower = lsqlin(C, d, A, b)

    valid3 = (NetPower > Coef.NominalPower * 0.75) & (NetPower < Coef.NominalPower * 1.05) & valid

    # #OCV: open circuit voltage
    if not Coef.NominalOCV:
        c2 = np.polyfit(Current[valid3], Voltage[valid3], 1) #linear fit of voltage vs. current
        Coef.NominalOCV = min(1.25, max(np.amax(Voltage), c2[-1]))#assure a positive OCV at Y-intercept of V-I relationship

    # ##Area
    if not Coef.Area:
        if not CellArea:
            if not Coef.NominalASR:
                if Type == 'SOFC':
                    Coef.NominalASR = 0.25
                elif Type == 'PAFC':
                    Coef.NominalASR = 0.5
                else:
                    Coef.NominalASR = 0.25

            Coef.Area = np.mean(Current * Coef.NominalASR / (Coef.NominalOCV-Voltage / Coef.Cells)) #effective area in cm^2
        else:
            Coef.Area = CellArea


    #ASR
    ASR = (Coef.NominalOCV - Voltage / Coef.Cells) / (Current / Coef.Area)
    nData = int(max(50, math.ceil(len(np.nonzero(valid3)[0]) / 20.0))) #1st 5# of data above 75% power, or 50 data points
    ASR = ASR[valid3]
    if not Coef.NominalASR:
        Coef.NominalASR = np.mean(ASR[:nData])


    # #linear degradation:
    if not Coef.LinearDegradation: #this is change in efficiency vs hours
        if Time is not None:
            deg = np.polyfit(Time[valid3], ASR, 1)#find the degradation in terms of how much less power you have from the same amount of fuel
            Coef.LinearDegradation = max(deg[0], 0)
        elif Type == 'SOFC':
            Coef.LinearDegradation = 4e-6
        elif Type == 'PAFC' or Type == 'MCFC':
            Coef.LinearDegradation = 4.5e-6
        else:
            Coef.LinearDegradation = 0

    # #Threshold degradation
    if not Coef.ThresholdDegradation: #hours before which there is no linear degradation
        if Time is not None: #find time when ASR is still 99# of nominal
            Time2 = Time[valid3]
            t = 1
            nS = len(Time2) - nData

            while t < nS and np.mean(ASR[t:t+nData]) < 1.1 * Coef.NominalASR:
                t = t+1

            Coef.ThresholdDegradation = Time2[t]

        elif Type == 'SOFC':
            Coef.ThresholdDegradation = 1e4
        elif Type == 'PAFC':
            Coef.ThresholdDegradation = 9e3
        else:
            Coef.ThresholdDegradation = 1e4

    #Restart degradation:
    if not Coef.ReStartDegradation:
        if Type == 'SOFC':
            Coef.ReStartDegradation = 1e-4
        elif Type == 'PAFC' or Type == 'MCFC':
            Coef.ReStartDegradation = 1e-3
        else:
            Coef.ReStartDegradation = 0

    return Coef


def FuelCell_Operate(Power,Tin,Starts,NetHours,Coef):
    ## Inputs:
    #Power (kW) requested power output
    #Tin (C) temperature of inlet air
    #Starts (#) cumulative # of starts on this system
    #NetHours (hrs) cumulative hours since last maintenance refurbishment
    #Coef (varies) a set of performance parameters describing this microturbine
    ## Outputs:
    #FuelFlow (mmBTU/hr) fuel demand of the microturbine to produce the required power
    #ExhaustFlow (kg/s) exhaust mass flow of the microturbine
    #ExhaustTemperature (C) exhaust temperature of microturbine
    #NetEfficiency (#) fuel-to-electric efficiency of the microturbine

    ## Calculations
    nPower = Power/Coef.NominalPower
    # ASR = Coef.NominalASR + Coef.ReStartDegradation*Starts + Coef.LinearDegradation * max(0, (NetHours - Coef.ThresholdDegradation)) #ASR  in Ohm*cm^2
    diff = NetHours - Coef.ThresholdDegradation
    diff[diff < 0] = 0
    ASR = Coef.NominalASR + Coef.ReStartDegradation*Starts + Coef.LinearDegradation * diff #ASR  in Ohm*cm^2
    Utilization = Coef.Utilization[0]*(1-nPower)**2 + Coef.Utilization[1]*(1-nPower) + Coef.Utilization[2] #decrease in utilization at part load
    Current = Power / Coef.Cells * 1000 / 0.8#initial guess assumes a voltage of 0.8 per cell
    HeatLoss = Power * Coef.StackHeatLoss
    AncillaryPower = 0.1 * Power#initial guess of 10#
    ##Total power = V*I = (OCV-ASR/area*I)*I:  find I & V
    for _ in range(4):
        Voltage = Coef.Cells * (Coef.NominalOCV - Current * ASR / Coef.Area)
        Current = 1.2 * (Power + AncillaryPower) * 1000.0 / Voltage - 0.2 * Current
        FuelFlow = Coef.Cells * Current *Coef.kg_per_Amp / Utilization
        ExhaustFlow = (Coef.Cells * Current * (1.2532 - Voltage / Coef.Cells) / 1000 - HeatLoss) / (1.144 * Coef.StackDeltaT) #flow rate in kg/s with a specific heat of 1.144kJ/kg*K
        AncillaryPower = Coef.AncillaryPower[0] * FuelFlow**2  + Coef.AncillaryPower[1] * FuelFlow + Coef.AncillaryPower[0]*ExhaustFlow**2 + Coef.AncillaryPower[1]*ExhaustFlow + Coef.AncillaryPower[2]*(Tin-18)*ExhaustFlow

    ExhaustTemperature = (FuelFlow * Coef.LHV - HeatLoss - Power)/(1.144*ExhaustFlow) + Tin + (Coef.ExhaustTemperature[0]*nPower**2 + Coef.ExhaustTemperature[1]*nPower + Coef.ExhaustTemperature[2])
    NetEfficiency = Power / (FuelFlow * Coef.LHV)
    FuelFlow = FuelFlow * Coef.LHV / 1.055 * 3600 / 1e6 #converting kg/s to mmBTU

    return FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency


for i in [1, 2, 3]:
    data = pd.read_csv("gen" + str(i) + ".csv", header=0)

    Power = data['Power'].values
    Fuel = data['Fuel'].values / 50144 # 50144 kJ / kg converts fuel kW to kg / s
    Time = data['Time'].values
    AmbTemperature = data['AmbTemperature'].values
    Start = data['Start'].values
    Hours = data['Hours'].values
    Valid = data['Valid'].values

    Coef = FuelCell_Calibrate('PAFC', 'CH4', Power, Fuel, Time, AmbTemperature, [], [], [], [], [], [], [], [], [])

    FuelFlow, ExhaustFlow, ExhaustTemperature, NetEfficiency = FuelCell_Operate(Power[Valid], AmbTemperature[Valid], Start[Valid], Hours[Valid], Coef)
    # What are we supposed to do with these results?
