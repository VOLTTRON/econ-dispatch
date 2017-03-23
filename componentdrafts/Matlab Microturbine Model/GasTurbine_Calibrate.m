function [Coef] = GasTurbine_Calibrate(Coef,Time,Power,Temperature,FuelFlow,AirFlow,ExhaustTemperature)
%This function determines the best fit coefficients for modeling a gas turbine
%% User can provide the following Coeficients, or they can be calculated here:
% 'NominalPower': The nominal power of the turbine
% 'Fuel_LHV': Lower heating value of the fuel in kJ/kg
% 'TempDerateThreshold': The temperature (C) at which the turbine efficiency starts to decline
% 'TempDerate': The rate at which efficiency declines (%/C) after the threshold temperature
% 'Maintenance': The rate of performance decline between maintenance cycles (%/yr)
%% Other inputs are:
% Time (cumulative hours of operation)
% Power (kW)
% Temperature (C) ambient
% FuelFlow (kg/s)
% AirFlow (kg/s) (Optional)
% Exhaust Temperature (C)  (Optional)

if isempty(Coef)
    Coef = [];
end
if ~isfield(Coef,'NominalPower')
    P2 = sort(Power);
    Coef.NominalPower = P2(ceil(.98*length(Power)));
end
if ~isfield(Coef,'Fuel_LHV')
    Coef.Fuel_LHV = 50144; % Assume natural gas with Lower heating value of CH4 in kJ/kg
end
Eff = Power./(FuelFlow*Coef.Fuel_LHV);
Eff(isnan(Eff)) = 0; %zero out any bad FuelFlow data

if ~isfield(Coef,'TempDerate')
    Tsort = sort(Temperature);
    Tmin = Tsort(ceil(0.1*length(Tsort)));
    Tmax = Tsort(ceil(0.9*length(Tsort)));
    C = zeros(10,1);
    for i = 1:1:10
        tl = Tmin + (i-1)/10*(Tmax-Tmin);
        th = Tmin + i/10*(Tmax-Tmin);
        valid = (Eff>0).*(Eff<0.5).*(Temperature>tl).*(Temperature<th);
        fit = polyfit(Temperature(valid),Eff(valid)*100,1);
        C(i) = -fit(1);
    end
    C(C<0.025) = 0; %negligable dependence on temperature
    I = find(C>0,1,'first');
    if isempty(I)
        Coef.TempDerateThreshold = 20;
        Coef.TempDerate = 0;
    else
        Coef.TempDerateThreshold = Tmin + I/10*(Tmax - Tmin);
        Coef.TempDerate = mean(C(C>0));
    end
end
if ~isfield(Coef,'Maintenance')
    valid = (Eff>0).*(Eff<0.5);
    fit = polyfit(Time(valid)/8760,Eff(valid)*100,1); %time in yrs, eff in %
    Coef.Maintenance = -fit(1);
end

Tderate = max(0,Coef.TempDerate.*(Temperature - Coef.TempDerateThreshold));
if ~isempty(Time)
    MaintenanceDerate = Time/8760.*Coef.Maintenance;
else MaintenanceDerate = 0;
end

Eff = Eff + Tderate/100 + MaintenanceDerate/100; %remove temperature dependence from data prior to fit
Coef.Eff = polyfit(Power/Coef.NominalPower,Eff,2);%efficiency of gas turbine before generator conversion
% Coef.Eff = polyfit(Power/Coef.NominalPower,Eff,3);

if ~isempty(AirFlow)
    Coef.FlowOut = polyfit(Power/Coef.NominalPower,AirFlow./FuelFlow,1);
    % Coef.FlowOut = polyfit(Power/Coef.NominalPower,AirFlow./FuelFlow,2);
elseif ~isempty(ExhaustTemperature) %assume a heat loss and calculate air mass flow
    AirFlow = (FuelFlow*Coef.Fuel_LHV - 1.667*Power)./(1.1*(ExhaustTemperature - Temperature));
    Coef.FlowOut = polyfit(Power/Coef.NominalPower,AirFlow./FuelFlow,1);
    % Coef.FlowOut = polyfit(Power/Coef.NominalPower,AirFlow./FuelFlow,2);
else %assume a flow rate
    Coef.FlowOut = 1;
end

if ~isempty(ExhaustTemperature)
    Coef.HeatLoss =  mean(((FuelFlow*Coef.Fuel_LHV - Power) - (ExhaustTemperature - Temperature)*1.1.*AirFlow)./Power); %flow rate in kg/s with a specific heat of 1.1kJ/kg*K
else
    Coef.HeatLoss = 2/3;
end