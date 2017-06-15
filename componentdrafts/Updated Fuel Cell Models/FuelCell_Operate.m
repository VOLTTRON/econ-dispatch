function [FuelFlow,ExhaustFlow,ExhaustTemperature,NetEfficiency] = FuelCell_Operate(Power,Tin,Starts,NetHours,Coef)
%% Inputs:
%Power (kW) requested power output
%Tin (C) temperature of inlet air
%Starts (#) cumulative # of starts on this system
%NetHours (hrs) cumulative hours since last maintenance refurbishment
%Coef (varies) a set of performance parameters describing this microturbine
%% Outputs:
%FuelFlow (mmBTU/hr) fuel demand of the microturbine to produce the required power
%ExhaustFlow (kg/s) exhaust mass flow of the microturbine
%ExhaustTemperature (C) exhaust temperature of microturbine
%NetEfficiency (%) fuel-to-electric efficiency of the microturbine

%% Calculations
nPower = Power/Coef.NominalPower;
ASR = Coef.NominalASR + Coef.ReStartDegradation*Starts + Coef.LinearDegradation*max(0,(NetHours-Coef.ThresholdDegradation)); %ASR  in Ohm*cm^2 
Utilization = Coef.Utilization(1)*(1-nPower).^2 + Coef.Utilization(2)*(1-nPower) + Coef.Utilization(3); %decrease in utilization at part load
Current = Power./Coef.Cells*1000/0.8;%initial guess assumes a voltage of 0.8 per cell
HeatLoss = Power*Coef.StackHeatLoss;
AncillaryPower = 0.1*Power;%initial guess of 10%
%%Total power = V*I = (OCV-ASR/area*I)*I:  find I & V
for j = 1:1:4
    Voltage = Coef.Cells*(Coef.NominalOCV - Current.*ASR/Coef.Area);
    Current = 1.2*(Power + AncillaryPower)*1000./Voltage - .2*Current;
    FuelFlow = Coef.Cells*Current*Coef.kg_per_Amp./Utilization;
    ExhaustFlow = (Coef.Cells*Current.*(1.2532 - Voltage/Coef.Cells)/1000 - HeatLoss)/(1.144*Coef.StackDeltaT); %flow rate in kg/s with a specific heat of 1.144kJ/kg*K
    AncillaryPower = Coef.AncillaryPower(1)*FuelFlow.^2  + Coef.AncillaryPower(2)*FuelFlow + Coef.AncillaryPower(1)*ExhaustFlow.^2 + Coef.AncillaryPower(2)*ExhaustFlow + Coef.AncillaryPower(3)*(Tin-18).*ExhaustFlow;
end
ExhaustTemperature = (FuelFlow*Coef.LHV - HeatLoss- Power)./(1.144*ExhaustFlow) + Tin + (Coef.ExhaustTemperature(1)*nPower.^2 + Coef.ExhaustTemperature(2)*nPower + Coef.ExhaustTemperature(3));
NetEfficiency = Power./(FuelFlow*Coef.LHV);
FuelFlow = FuelFlow*Coef.LHV/1.055*3600/1e6; %converting kg/s to mmBTU

%% default parameters
%%SOFC
% Coef.NominalPower =100;
% Coef.NominalASR = .25;
% Coef.ReStartDegradation = 1e-4;
% Coef.LinearDegradation = 4e-6;
% Coef.ThresholdDegradation = 1e4; %hours before which there is no linear degradation
% Coef.Utilization = [-.2, -.2 ,.8];
% Coef.StackHeatLoss = .1;
% Coef.Area = 1250; %1250cm^2 and 100 cells producing 100kW works out to 0.8 W/cm^2 and at a voltage of .8 this is 1 amp/cm^2
% Coef.Cells = 100;
% Coef.NominalOCV = 1.1;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [2, 4, 5];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.LHV = 50144;
% Coef.kg_per_Amp = 2/1000/96485;

%%PAFC
% Coef.NominalPower =200;
% Coef.NominalASR = .5; 
% Coef.ReStartDegradation = 1e-3;
% Coef.LinearDegradation = 4.5e-6;
% Coef.ThresholdDegradation = 9e3; %hours before which there is no linear degradation
% Coef.Utilization = [-.3, -.2 ,.66]; % calibrated examples:[-.12, -.5 ,.685; -.18, -.32 ,.712;  -.11, -.30 ,.66];
% Coef.StackHeatLoss = .1;
% Coef.Area = 5000; %5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
% Coef.Cells = 200;
% Coef.NominalOCV = 0.8;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [.5, 4, .2];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.LHV = 50144;
% Coef.kg_per_Amp = 2/1000/96485;