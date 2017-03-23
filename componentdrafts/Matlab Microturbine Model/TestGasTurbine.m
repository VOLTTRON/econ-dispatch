%Coeficients are for a Capstone C60 gas turbine
Coef.NominalPower = 60;
Coef.Fuel_LHV = 50144;
Coef.TempDerateThreshold = 15.556;%from product specification sheet
Coef.TempDerate = 0.12;% (%/C) from product specification sheet
Coef.Maintenance = 3;% in percent/year
Coef.HeatLoss = 2/3;
Coef.Eff = [-0.2065, 0.3793, 0.1043];
Coef.FlowOut = [-65.85,164.5];

%% illustrates the calibration of the GasTurbine function
load('CapstoneTurndownData.mat');
T = DataC60(:,3) - 273;
Coef = GasTurbine_Calibrate(Coef,[],DataC60(:,2),T,DataC60(:,4),DataC60(:,5),[]);
%%%---%%%

Pdemand = DataC60(:,2);
[AirFlow,FuelFlow,Tout,Efficiency] = GasTurbine_Operate(Pdemand,T,0,Coef);
Efficiency = Efficiency*100;
Rsquared = 1-sum((DataC60(:,6) - Efficiency).^2)./sum((DataC60(:,6) - mean(DataC60(:,6))).^2);

Rsquared2 = 1-sum((DataC60(:,5) - AirFlow).^2)./sum((DataC60(:,5) - mean(DataC60(:,5))).^2);

figure(5)
plot(DataC60(:,2),Efficiency);
hold on
scatter(DataC60(:,2),DataC60(:,6));
legend('Fit','Data')
xlabel('Power Output','FontSize',12)
ylabel('Efficiency (%)','FontSize',12)

figure(6)
plot(DataC60(:,2),AirFlow);
hold on
scatter(DataC60(:,2),DataC60(:,5));
legend('Fit','Data')
xlabel('Power Output','FontSize',12)
ylabel('Air Flow Rate (kg/s)','FontSize',12)

figure(7)
plot(DataC60(:,2),FuelFlow);
hold on
scatter(DataC60(:,2),DataC60(:,4));
legend('Fit','Data')
xlabel('Power Output','FontSize',12)
ylabel('Fuel Flow Rate (kg/s)','FontSize',12)

figure(8)
plot(DataC60(:,2),Tout);
legend('Fit')
