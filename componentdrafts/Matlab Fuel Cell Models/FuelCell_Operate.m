function [FuelFlow,ExhaustFlow,ExhaustTemperature,NetEfficiency] = FuelCell_Operate(Power,Tin,Starts,NetHours,Coef)
F = 96485; %Faraday constant C/mol
switch Coef.Fuel
    case {'CH4'}
        n = 8; % # of electrons per molecule (assuming conversion to H2)
        LHV = 50144; %Lower heating value of CH4 in kJ/g
        m_fuel = 16;%molar mass
    case {'H2'}
        n = 2; % # of electrons per molecule (assuming conversion to H2)
        LHV = 120210; %Lower heating value of H2 in kJ/kmol
        m_fuel = 2;%molar mass
end

nPower = Power/Coef.NominalPower;
ASR = Coef.NominalASR + Coef.ReStartDegradation*Starts + Coef.LinearDegradation*max(0,(NetHours-Coef.ThresholdDegradation)); %ASR  in Ohm*cm^2 
Utilization = Coef.Utilization(1)*(1-nPower).^2 + Coef.Utilization(2)*(1-nPower) + Coef.Utilization(3); %decrease in utilization at part load
Current = Coef.Area*(Coef.NominalCurrent(1)*nPower.^2 + Coef.NominalCurrent(2)*nPower + Coef.NominalCurrent(3)); %first guess of current
HeatLoss = Power*Coef.StackHeatLoss;
AncillaryPower = 0.1*Power;
for j = 1:1:4
    Voltage = Coef.Cells*(Coef.NominalOCV - Current.*ASR/Coef.Area);
    Current = Coef.gain*(Power + AncillaryPower)*1000./Voltage - (Coef.gain-1)*Current;
    FuelFlow = m_fuel*Coef.Cells*Current./(n*1000*F*Utilization);
    ExhaustFlow = (Coef.Cells*Current.*(1.2532 - Voltage/Coef.Cells)/1000 - HeatLoss)/(1.144*Coef.StackDeltaT); %flow rate in kg/s with a specific heat of 1.144kJ/kg*K
    AncillaryPower = Coef.AncillaryPower(1)*FuelFlow.^2  + Coef.AncillaryPower(2)*FuelFlow + Coef.AncillaryPower(1)*ExhaustFlow.^2 + Coef.AncillaryPower(2)*ExhaustFlow + Coef.AncillaryPower(3)*(Tin-18).*ExhaustFlow;
end
ExhaustTemperature = ((Coef.Cells*Current.*(1.2532 - Voltage/Coef.Cells)/1000 - HeatLoss) + (1-Utilization).*FuelFlow*LHV)./(1.144*ExhaustFlow) + Tin + (Coef.ExhaustTemperature(1)*nPower.^2 + Coef.ExhaustTemperature(2)*nPower + Coef.ExhaustTemperature(3));
NetEfficiency = Power./(FuelFlow*LHV);

%% default parameters
%%SOFC
% Coef.Fuel = 'CH4';
% Coef.NominalPower =100;
% Coef.NominalASR = .25;
% Coef.ReStartDegradation = 1e-4;
% Coef.LinearDegradation = 4e-6;
% Coef.ThresholdDegradation = 1e4; %hours before which there is no linear degradation
% Coef.Utilization = [-.2, -.2 ,.8];
% Coef.NominalCurrent = [0.2 0.5 0];
% Coef.StackHeatLoss = .1;
% Coef.Area = 1250; %1250cm^2 and 100 cells producing 100kW works out to 0.8 W/cm^2 and at a voltage of .8 this is 1 amp/cm^2
% Coef.Cells = 100;
% Coef.NominalOCV = 1.1;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [2, 4, 5];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.gain = 1.2;

%%PAFC
% Coef.Fuel = 'CH4';
% Coef.NominalPower =200;
% Coef.NominalASR = .5; 
% Coef.ReStartDegradation = 1e-3;
% Coef.LinearDegradation = 4.5e-6;
% Coef.ThresholdDegradation = 9e3; %hours before which there is no linear degradation
% Coef.Utilization = [-.3, -.2 ,.66]; % calibrated examples:[-.12, -.5 ,.685; -.18, -.32 ,.712;  -.11, -.30 ,.66];
% Coef.NominalCurrent = [0.1921 0.1582 0.0261];
% Coef.StackHeatLoss = .1;
% Coef.Area = 5000; %5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
% Coef.Cells = 200;
% Coef.NominalOCV = 0.8;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [.5, 4, .2];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.gain = 1.4;