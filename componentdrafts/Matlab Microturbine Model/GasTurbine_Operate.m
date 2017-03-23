function [AirFlow,FuelFlow,Tout,Efficiency] = GasTurbine_Operate(Power,Tin,NetHours,Coef)
%Pdemand in kW
%Tin in C
%Net Hours in hours since last maintenance event
Tderate = max(0,Coef.TempDerate/100.*(Tin - Coef.TempDerateThreshold)); %efficiency scales linearly with temp
MaintenanceDerate = NetHours/8760.*Coef.Maintenance/100;%efficiency scales linearly with hours since last maintenance.
Pnorm = (Power/Coef.NominalPower);

Efficiency = (Coef.Eff(1).*Pnorm.^2 + Coef.Eff(2).*Pnorm + Coef.Eff(3)) - Tderate - MaintenanceDerate;
% Efficiency = (Coef.Eff(1).*Pnorm.^3 + Coef.Eff(2).*Pnorm.^2 + Coef.Eff(3).*Pnorm + Coef.Eff(4)) - Tderate - MaintDerate;
FuelFlow = Power./(Efficiency*Coef.Fuel_LHV);
AirFlow = FuelFlow.*(Coef.FlowOut(1).*Pnorm + Coef.FlowOut(2)); %air mass flow rate in kg/s
% AirFlow = FuelFlow.*(Coef.FlowOut(1).*Pnorm.^2 + Coef.FlowOut(2).*Pnorm + Coef.FlowOut(3)); %air mass flow rate in kg/s
Tout = Tin + (FuelFlow*Coef.Fuel_LHV - (1+Coef.HeatLoss)*Power)./(1.1*AirFlow); %flow rate in kg/s with a specific heat of 1.1kJ/kg*K