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

% figure(5)
% plot(DataC60(:,2),Efficiency);
% hold on
% scatter(DataC60(:,2),DataC60(:,6));
% legend('Fit','Data')
% xlabel('Power Output','FontSize',12)
% ylabel('Efficiency (%)','FontSize',12)
% 
% figure(6)
% plot(DataC60(:,2),AirFlow);
% hold on
% scatter(DataC60(:,2),DataC60(:,5));
% legend('Fit','Data')
% xlabel('Power Output','FontSize',12)
% ylabel('Air Flow Rate (kg/s)','FontSize',12)

% figure(7)
% plot(DataC60(:,2),FuelFlow);
% hold on
% scatter(DataC60(:,2),DataC60(:,4));
% legend('Fit','Data')
% xlabel('Power Output','FontSize',12)
% ylabel('Fuel Flow Rate (kg/s)','FontSize',12)
% 
% figure(8)
% plot(DataC60(:,2),Tout);
% legend('Fit')

Xdata = DataC60(:,2);
Ydata = FuelFlow*171.11;  % kg/s -> mmBtu

N = length(Xdata);
n1 = find(Xdata<62,1,'last');
n2 = find(Xdata<62,1,'last');
% n1 = find(DataC60(:,2)<20,1,'last');
% n2 = find(DataC60(:,2)<44,1,'last');
xmin_Turbine = zeros(1,1);
xmax_Turbine = zeros(1,1);
Xdata = [ones(N,1), Xdata];
m1 = regress(Ydata(1:n1), Xdata(1:n1,:));
m2 = regress(Ydata(n1:n2), Xdata(n1:n2,:));
m3 = regress(Ydata(n2:end), Xdata(n2:end,:));
m_Turbine = m1;
% m_GasTurbine = [m1,m2,m3];
xmax_Turbine(1) = max(Xdata(1:n1,2));
xmin_Turbine(1) = min(Xdata(1:n1,2));
% xmax_GasTurbine(2) = max(DataC60(n1:n2,2));
% xmin_GasTurbine(2) = xmax_GasTurbine(1);
% xmax_GasTurbine(3) = max(DataC60(n2:end,2));
% xmin_GasTurbine(3) = xmax_GasTurbine(2);

figure(1)
scatter(Xdata(:,2),Ydata);
xlabel('Power Output (kW)')
ylabel('Fuel Flow Rate (mmBTU/hr)')
% ylabel('Fuel Flow Rate (mmBtu/hr)')
hold on
for i=1:1
    x = linspace(xmin_Turbine(i), xmax_Turbine(i), 100);
    y = m_Turbine(1,i)+m_Turbine(2,i)*x;
    plot(x,y,'r','LineWidth',3)
end
hold off

save('GasTurbinePara.mat','xmin_Turbine','xmax_Turbine','m_Turbine');
