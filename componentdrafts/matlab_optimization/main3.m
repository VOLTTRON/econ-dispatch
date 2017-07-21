%{
clear
close all

dates.train_start = '06/01 00:00:00';
dates.train_stop = '06/26 24:00:00';
dates.test_start = '08/09 00:00:00';
dates.test_stop = '08/09 24:00:00'; %% Cannot be a weekend day

heatCoolElec = 1; %1,2,3

[tbl, isTestData] = LoadData2('McClureAltBaselineLoads.csv', dates, heatCoolElec);
trInds = find(~isTestData);
teInds = find(isTestData);
trainTbl = tbl(trInds, :);
testTbl = tbl(teInds, :);

model = fitrtree(trainTbl,'Energy','categoricalpredictors', {'Weekday','Hour'});
Power_pred = predict(model, testTbl);
% Power_pred = smooth(Power_pred,3);

Power_actual = tbl.Energy(teInds);
RMSE0 = sqrt(mse(Power_actual,Power_pred));

figure
plot(0:23, Power_actual, 0:23, Power_pred)
legend('Actual','Forecast')
title(['RMSE: ',num2str(RMSE0)])
ylabel('Main Meter (kW)')
xlabel('Time (hour)')

%}
clear 

% get the electricity and thermal load forecasts
data = xlsread('Hospital Modeled Data.xlsx');
range = 4226:4249; % 6/26 both cooling and heating

E_PV = zeros(24,1);     %kW
E_load = data(range,2);    % kW
Q_loadheat = data(range,3)/293.1;  % kW -> mmBtu/hr
Q_loadcool = data(range,4)/293.1;  % kW -> mmBtu/hr

% get the model parameters and bounds for variables
load FuelCellPara.mat
load BoilerPara.mat
load ChillerIGVPara.mat
% load ChillerVSDPara.mat
load AbsChillerPara.mat

% get the price info
lambda_gas = 7.614*ones(24,1);   %$/mmBtu, original price 7.614
lambda_elec_fromgrid = 0.1*ones(24,1);  %$/kWh, original price 0.1
lambda_elec_togrid = 0.1*ones(24,1);  %$/kWh, original price 0.1

% component capacity 
cap_FuelCell = 500; % kW
cap_abs = 464/293.1; % kW -> mmBtu/hr
cap_boiler = 8; % mmBtu/hr
Nchiller = 4;
cap_chiller = 200*3.517/293.1; % ton -> mmBtu/hr

%% compute the parameters for the optimization
% boiler
xmin_Boiler(1) = cap_boiler*0.2; % !!!need to consider cases when xmin is not in the first section of the training data
Nsection = find(xmax_Boiler>cap_boiler,1,'first');
xmin_Boiler = xmin_Boiler(1:Nsection);
xmax_Boiler = xmax_Boiler(1:Nsection);
a_boiler = m_Boiler(2,1:Nsection); 
b_boiler = m_Boiler(1,1)+a_boiler(1)*xmin_Boiler(1);
xmax_Boiler(end) = cap_boiler;

% absorption chiller
flagabs = 1;
xmin_AbsChiller(1) = cap_abs*0.2;
a_abs = m_AbsChiller(2);    
b_abs = m_AbsChiller(1)+a_abs*xmin_AbsChiller(1);   
xmax_AbsChiller(end) = cap_abs;

% chiller
for i=1:Nchiller
    xmin_Chiller{i}(1) = cap_chiller*0.15;
    a_chiller{i} = m_ChillerIGV(2)+(i-1)*0.01;  % adding 0.01 to the slopes to differentiate the chillers
    b_chiller{i} = m_ChillerIGV(1)+a_chiller{i}(1)*xmin_ChillerIGV(1);
    xmax_Chiller{i} = cap_chiller;
end

% generator
xmin_Turbine(1) = cap_FuelCell*0.3;
a_E_turbine = m_Turbine(2);  
b_E_turbine = m_Turbine(1)+a_E_turbine*xmin_Turbine(1); 
xmax_Turbine(end) = cap_FuelCell;

a_Q_turbine = a_E_turbine-1/293.1;  
b_Q_turbine = b_E_turbine-xmin_Turbine(1)/293.1;

% heat recovery unit
a_hru = 0.8;

%% 
save('input4.mat','lambda_gas','lambda_elec_togrid','lambda_elec_fromgrid', ...
    'E_PV','E_load','Q_loadheat','Q_loadcool','a_hru', ...
    'xmax_Turbine','xmin_Turbine','a_E_turbine','b_E_turbine','a_Q_turbine','b_Q_turbine', ...
    'xmax_Boiler','xmin_Boiler','a_boiler','b_boiler', ...
    'flagabs','xmax_AbsChiller','xmin_AbsChiller','a_abs','b_abs', ...
    'Nchiller','xmax_Chiller','xmin_Chiller','a_chiller','b_chiller');

%{
figure
subplot(2,3,1)
for i=1:Nsection
x = linspace(xmin_Boiler(i),xmax_Boiler(i),100);
y = m_Boiler(1,i)+m_Boiler(2,i)*x;
plot(x,y,'k','LineWidth',2)
hold on
end
hold off
xlabel('Heat Output (mmBTU/hr)')
ylabel('Gas Heat Input (mmBTU/hr)')
title('Boiler')
axis tight
    
%}