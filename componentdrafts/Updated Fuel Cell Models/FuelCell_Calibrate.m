function Coef = FuelCell_Calibrate(Type,Fuel,NetPower,FuelFlow,Time,InletTemperature,AirFlow,ExhaustTemperature,Voltage,Current,AncillaryPower,Cells,CellArea,StackDeltaT,Coef)
%% Inputs
%Type: string of what kind of FC 'PEM', 'PAFC', 'MCFC', or 'SOFC' (NOT optional)
%Fuel: string of what type of fuel is provided 'CH4' or 'H2' (NOT optional)
%NetPower: power in kW (NOT optional)
%FuelFlow: in kg/s  (NOT optional)
%Time: cumulative hours that the FC has been running (NOT optional)
%InletTemperature (ambient conditions): in Celsius (NOT optional)
%AirFlow: in kg/s 
%ExhaustTemperature: in Celsius  
%Voltage: Stack voltage in V
%Current: Cumulative current (if there are multiple stacks) in Amps
%AncillaryPower: Power internally consumed by blowers pumps and fans (kW)
%CellTemperature: nominal operating temperature in Celsius
%Cells: number of cells in the stack
%CellArea: effective cell area in cm^2: Power (kW) = current density (A/cm^2)* Area (cm^2) * Voltage (V) / 1000
%StackDeltaT: nominal temperature difference accross the stack in degrees Celsius
%% Outputs:
%Coef: structure of previously or user defined coefficients, default isempty [];

%%---%%%
%% Calculations
%Fueltype
if ~isfield(Coef,'Fuel')
    if ~isempty(Fuel)
        Coef.Fuel = Fuel;
    else
        Coef.Fuel = 'CH4';
    end
end
switch Coef.Fuel
    case {'CH4'}
        n = 8; % # of electrons per molecule (assuming conversion to H2)
        LHV = 50144; %Lower heating value of CH4 in kJ/kg
        m_fuel = 16;%molar mass kg/kmol
    case {'H2'}
        n = 2; % # of electrons per molecule (assuming conversion to H2)
        LHV = 120210; %Lower heating value of H2 in kJ/kg
        m_fuel = 2;%molar mass kg/kmol
end
Coef.kg_per_Amp = m_fuel/(n*1000*96485);
Coef.LHV = LHV;
E1 = 1000*Coef.LHV*Coef.kg_per_Amp;%voltage potential of fuel (efficiency = utilization*voltage/E1)

%Power
NetPower(NetPower<0) = 0;
sortPow = sort(NetPower);%NetPower
if ~isfield(Coef,'NominalPower')
    Coef.NominalPower = round(sortPow(round(length(sortPow)*.98))); %assume 2% outliers in data
end
nPower = NetPower/Coef.NominalPower;
Efficiency = NetPower./(FuelFlow*LHV);
Efficiency(isnan(Efficiency)) = 0; Efficiency(isinf(Efficiency)) = 0;
valid = (Efficiency>.01)& (Efficiency<0.7);
a = polyfit(NetPower(valid),Efficiency(valid),2);
valid = (Efficiency>0.8*(a(1)*NetPower.^2 + a(2)*NetPower + a(3)))& (Efficiency<1.2*(a(1)*NetPower.^2 + a(2)*NetPower + a(3)));
nominal = (NetPower>Coef.NominalPower*0.95) & (NetPower<Coef.NominalPower*1.05) & valid;

%stack temperature gradient
if ~isfield(Coef,'StackDeltaT')
    if ~isempty(StackDeltaT)
        Coef.StackDeltaT = StackDeltaT;
    else
        Coef.StackDeltaT = 100;
    end
end

%# of cells
if ~isfield(Coef, 'Cells')
    if ~isempty(Cells)
        Coef.Cells = Cells;
    elseif  ~isempty(Voltage)
        util = 0.8 - (0.6-mean(Efficiency(nominal)));%estimate fuel utilization at peak efficiency
        Coef.Cells = round(mean(util*Voltage(nominal)./(Efficiency(nominal)*E1))); %estimate of the # of cells: Eff = util*V/Cells/E1
    elseif ~isempty(Current)
        Utilization = Efficiency(nominal).^.5; %assumed fuel utilization
        Coef.Cells = round(mean((FuelFlow(nominal)/m_fuel).*Utilization*n*1000*96485./Current(nominal)));
    else
        Coef.Cells = round(Coef.NominalPower);%Assumption 1kW cells
    end
end


%%At least 2 of 3 must be known (Current, Voltage, & Ancillary Power), or some assumptions will be made: 
if ~isempty(Voltage) && ~isempty(Current)
    % if voltage & current are known, ancillary power is easily computed.  
    TotalPower = Voltage*Current/1000; %stack Power in kW
    AncillaryPower = TotalPower - NetPower; %first guess of ancillary power for actual operation
elseif isempty(AncillaryPower)
    %estimate an air flow
    if isempty(AirFlow)
        ExhaustFlow = (FuelFlow*LHV - NetPower)/(1.144*2*Coef.StackDeltaT); %flow rate in kg/s with a specific heat of 1.144kJ/kg*K
    else ExhaustFlow = AirFlow;
    end
    nominalEfficiency = mean(Efficiency(nominal));
    a = polyfit(InletTemperature(nominal)-18,FuelFlow(nominal)*LHV*nominalEfficiency-Coef.NominalPower,1);
    Coef.AncillaryPower(3) = 0.5*a(1)/mean(ExhaustFlow(nominal));
    Coef.AncillaryPower(2) = (0.05*Coef.NominalPower - 0.5*a(1)*mean(InletTemperature(nominal)-18))/mean(ExhaustFlow(nominal)+FuelFlow(nominal));%makes it so nominal ancillary power is 15%
    AncillaryPower = Coef.AncillaryPower(1)*FuelFlow.^2  + Coef.AncillaryPower(2)*FuelFlow + Coef.AncillaryPower(1)*ExhaustFlow.^2 + Coef.AncillaryPower(2)*ExhaustFlow + Coef.AncillaryPower(3)*(InletTemperature-18).*ExhaustFlow;
end
TotalPower = AncillaryPower+NetPower;
if ~isempty(Voltage)
    %if we know voltage & ancillary power we can find current
    Current = TotalPower*1000./Voltage;
elseif ~isempty(Current)
    %if we know current& ancillary power we can find voltage    
    Voltage = TotalPower*1000./Current;
else
    %otherwise cell area, OCV & ASR are assumed
    %efficiency = (current*voltage - ancillary power)/energy in
    %Voltage = OCV - ASR*I/area
    %these two expressions reduce to a single quadratic of I that must be solved
    if ~isfield(Coef, 'NominalASR')
        if strcmp(Type,'SOFC')
            Coef.NominalASR = 0.25;
        elseif strcmp(Type,'PAFC')
            Coef.NominalASR = 0.5;
        elseif strcmp(Type,'MCFC')
            Coef.NominalASR = 0.75;
        elseif strcmp(Type,'PEM')
            Coef.NominalASR = 0.2;
        else
            Coef.NominalASR = 0.25;
        end
    end
    if ~isfield(Coef,'NominalOCV')
        if strcmp(Type,'SOFC')
            Coef.NominalOCV = 1;
        elseif strcmp(Type,'PAFC')
            Coef.NominalOCV = 0.8;
        elseif strcmp(Type,'MCFC')
            Coef.NominalOCV = 0.85;
        elseif strcmp(Type,'PEM')
            Coef.NominalOCV = 0.75;
        else
            Coef.NominalOCV = 0.9;
        end
    end
    nData = find(nominal,max(50,ceil(nnz(nominal)/20)));%1st 5% of data above 75% power, or 50 data points
    nominal5perc = nominal & ((1:length(nominal))'<nData(end));
    %given OCV & ASR, can find nominal utilization, voltage & current resulting in correct efficiency
    nominalCurrent = mean(TotalPower(nominal5perc))*1000./(0.8*Coef.NominalOCV*Coef.Cells);
    if ~isfield(Coef, 'Area')
        if isempty(CellArea)
            if strcmp(Type,'SOFC')
                Coef.Area = nominalCurrent/0.5; %assume a current density of 0.5A/cm^2
            elseif strcmp(Type,'PAFC')
                Coef.Area = nominalCurrent/0.33; %assume a current density of 0.25A/cm^2
            elseif strcmp(Type,'MCFC')
                Coef.Area = nominalCurrent/0.25; %assume a current density of 0.2A/cm^2
            elseif strcmp(Type,'PEM')
                Coef.Area = nominalCurrent/0.5; %assume a current density of 0.5A/cm^2
            else
                Coef.Area = nominalCurrent/0.5; %assume a current density of 0.5A/cm^2
            end
        else
            Coef.Area = CellArea;
        end
    end
    for j = 1:1:4 %find the V & I assuming this OCV and ASR
        nominalVoltage = Coef.Cells*(Coef.NominalOCV - nominalCurrent*Coef.NominalASR/Coef.Area);
        nominalCurrent = 1.2*mean(TotalPower(nominal5perc))*1000./nominalVoltage - .2*nominalCurrent;
    end
    nominalUtil = nominalCurrent*Coef.Cells/(mean(FuelFlow(nominal5perc))/m_fuel)/(n*1000*96485);
    valid2 = (NetPower>Coef.NominalPower*0.15) & (NetPower<Coef.NominalPower*0.95) & valid;
    nData = find(valid2,max(50,ceil(nnz(valid2)/5)));%1st 5% of data above 75% power, or 50 data points
    valid2 = valid2 & ((1:length(valid2))'<nData(end));   
    %%Total power = V*I = (OCV-ASR/area*I)*I:  find I & V
    Current = TotalPower(valid2)*1000/nominalVoltage;
    for j = 1:1:4 %find the V & I assuming this OCV and ASR
        Voltage = Coef.Cells*(Coef.NominalOCV - Current*Coef.NominalASR/Coef.Area);
        Current = 1.2*TotalPower(valid2)*1000./Voltage - .2*Current;
    end
    Utilization = Current*Coef.Cells/(n*1000*96485)./(FuelFlow(valid2)/m_fuel);
    pow = Voltage.*Current/1000;
    for j = 1:1:length(Utilization)% fit the maximum utilizations at each power level (steady-state)
        range = (pow>.99*pow(j)) & (pow<1.01*pow(j));
        if Utilization(j)<max(Utilization(range))
            Utilization(j) = 0;
        end
    end
    % estimate the controller part-load Utilization vs. Power initially when ASR = nominal ASR
    npow = nPower(valid2);
    npow = npow(Utilization>0);
    Utilization = Utilization(Utilization>0);
    C = [(1-npow).^2, (1-npow), ones(length(npow),1)];
    d = Utilization;
    Aeq = [0 0 1]; beq = nominalUtil;
    A = [1 0 0; 0 1 0; -1 1 0]; b = [0;0;0];
    Coef.Utilization = lsqlin(C,d,A,b,Aeq,beq);

%         %%Plot for verification
%         figure(19)
%         scatter(npow,Utilization)
%         hold on
%         X = linspace(0,1);
%         Y = Coef.Utilization(1)*(1-X).^2 + Coef.Utilization(2)*(1-X) + Coef.Utilization(3);
%         plot(X,Y,'g')

    % recalculate V & I as ASR degrades so that efficiency matches
    %current = reverse function of Utilization: ,Power & fuel flow are inputs
    Utilization = Coef.Utilization(1)*(1-nPower).^2 + Coef.Utilization(2)*(1-nPower) + Coef.Utilization(3); %decrease in utilization at part load
    Current = Utilization.*FuelFlow/m_fuel*(n*1000*96485)/Coef.Cells;
    
    %voltage is whatever is necessary to produce correct total power
    % degradding ASR is this evolving relationshp between current and voltage
    Voltage = TotalPower*1000./Current;
    Voltage(isinf(Voltage)) = 0;
    Voltage(isnan(Voltage)) = 0;

%     %%Plot for validation
%     ASR = (Coef.NominalOCV - Voltage/Coef.Cells)*Coef.Area./Current;
%     deg = polyfit((1:nnz(nominal))',ASR(nominal),1);
%     deg(1);
%     figure(9)
%     plot(ASR(nominal))
end

%Part-Load Utilization is calculated from fuel flow and current.
if ~isfield(Coef, 'Utilization')
    Utilization = Current*Coef.Cells/(n*1000*96485)./(FuelFlow/m_fuel);
    valid2 = (NetPower>Coef.NominalPower*0.15) & (NetPower<Coef.NominalPower*0.9) & valid;
    nData = find(valid2,max(50,ceil(nnz(valid2)/5)));%1st 5% of data above 75% power, or 50 data points
    valid2 = valid2 & ((1:length(valid2))'<nData(end));
    % estimate the controller part-load Utilization vs. Power initially when ASR = nominal ASR
    C = [(1-nPower(valid2)).^2, (1-nPower(valid2)), ones(nnz(valid2),1)];
    d = Utilization(valid2);
    A = [1 0 0; 0 1 0;0 0 -1]; b = [0;0;-.9*util];
    Coef.Utilization = lsqlin(C,d,A,b);
end

%Heat Loss
if ~isfield(Coef,'StackHeatLoss')
    if isempty(AirFlow)
        Coef.StackHeatLoss = 0.1; %assume 10% heat loss
    else
        Qgen = Coef.Cells*(1.2532 - Voltage/Coef.Cells).*Current/1000;
        Coef.StackHeatLoss = mean(Qgen - AirFlow*1.144*Coef.StackDeltaT)/Coef.NominalPower; %%flow rate in kg/s with a specific heat of 1.144kJ/kg*K
    end
end

%Air flow from the heat gen - heat loss / deltaT,
if isempty(AirFlow)
    AirFlow = (Coef.Cells*Current.*(1.2532 - Voltage/Coef.Cells)/1000 - Coef.StackHeatLoss*NetPower)/(1.144*Coef.StackDeltaT); %flow rate in kg/s with a specific heat of 1.144kJ/kg*K
end

% adjust errors in energy balance calculations to measured exhaust temperature
if ~isfield(Coef,'ExhaustTemperature')%deviation from calculated exhaust temp
    if ~isempty(ExhaustTemperature)
        if isempty(InletTemperature)
            InletTemperature = 300;
        end
        CalculatedExhaustTemperature = (FuelFlow*LHV - NetPower - Coef.StackHeatLoss*Coef.NominalPower)./(1.144*AirFlow) + InletTemperature;
        Coef.ExhaustTemperature = polyfit(nPower,ExhaustTemperature - CalculatedExhaustTemperature,2); %adjustment to calculated exhaust temperature
    else
        Coef.ExhaustTemperature = [0,0,0];
    end
end

%AncillaryPower
if ~isfield(Coef,'AncillaryPower')
    C = [(FuelFlow(nominal)+AirFlow(nominal)).^2, (FuelFlow(nominal)+AirFlow(nominal)), (InletTemperature(nominal)-18)];
    d = AncillaryPower;
    A = [-1 0 0; 0 -1 0; 0 0 -1]; b = [0;0;0];
    Coef.AncillaryPower = lsqlin(C,d,A,b);
end

valid3 = (NetPower>Coef.NominalPower*0.75) & (NetPower<Coef.NominalPower*1.05) & valid;
%OCV: open circuit voltage
if ~isfield(Coef,'NominalOCV')
    c2 = polyfit(Current(valid3),Voltage(valid3),1); %linear fit of voltage vs. current
    Coef.NominalOCV = min(1.25,max(max(Voltage),c2(end)));%assure a positive OCV at Y-intercept of V-I relationship
end

%%Area
if ~isfield(Coef, 'Area')
    if isempty(CellArea)
        if ~isfield(Coef, 'NominalASR')
            if strcmp(Type,'SOFC')
                Coef.NominalASR = 0.25;
            elseif strcmp(Type,'PAFC')
                Coef.NominalASR = 0.5;
            else
                Coef.NominalASR = 0.25;
            end
        end
        Coef.Area = mean(Current*Coef.NominalASR./(Coef.NominalOCV-Voltage/Coef.Cells)); %effective area in cm^2
    else
        Coef.Area = CellArea;
    end
end

%ASR
ASR = (Coef.NominalOCV-Voltage/Coef.Cells)./(Current/Coef.Area);
nData = max(50,ceil(nnz(valid3)/20));%1st 5% of data above 75% power, or 50 data points
ASR = ASR(valid3);
if ~isfield(Coef, 'NominalASR')
    Coef.NominalASR = mean(ASR(1:nData));
end

%linear degradation:
if ~isfield(Coef, 'LinearDegradation')%this is change in efficiency vs hours
    if~isempty(Time)
        deg = polyfit(Time(valid3),ASR,1);%find the degradation in terms of how much less power you have from the same amount of fuel
        Coef.LinearDegradation = max(deg(1),0);
    elseif strcmp(Type,'SOFC')
        Coef.LinearDegradation = 4e-6;
    elseif strcmp(Type,'PAFC')||strcmp(Type,'MCFC')
        Coef.LinearDegradation = 4.5e-6;
    else Coef.LinearDegradation = 0;
    end
end

%Threshold degradation
if ~isfield(Coef, 'ThresholdDegradation')%hours before which there is no linear degradation
    if~isempty(Time) %find time when ASR is still 99% of nominal
        Time2 = Time(valid3);
        t = 1;
        nS = length(Time2)-nData;
        while t<nS && mean(ASR(t:t+nData))<1.1*Coef.NominalASR;
            t = t+1;
        end
        Coef.ThresholdDegradation = Time2(t);
    elseif strcmp(Type,'SOFC')
        Coef.ThresholdDegradation = 1e4;
    elseif strcmp(Type,'PAFC')
        Coef.ThresholdDegradation = 9e3;
    else Coef.ThresholdDegradation = 1e4; 
    end
end

%Restart degradation:
if ~isfield(Coef,'ReStartDegradation')
%     starts = find((NetPower(2:end)>0.05*Coef.NominalPower).*(NetPower(1:end-1)<0.05*Coef.NominalPower));%indices of re-start moments
%     if length(starts)>5
%         ASRlast5 = mean(ASR(end-nData:end));
%         Coef.ReStartDegradation = max(0,(ASRlast5-Coef.NominalASR+Time(end)*Coef.LinearDegradation)/length(starts));
%     else
        if strcmp(Type,'SOFC')
            Coef.ReStartDegradation = 1e-4;
        elseif strcmp(Type,'PAFC') ||strcmp(Type,'MCFC')
            Coef.ReStartDegradation = 1e-3;
        else Coef.ReStartDegradation = 0;
        end
%     end
end
