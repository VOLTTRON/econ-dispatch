clc
clear all
%*************     Deployment Module     ********
%**************     Screw Chiller      **********
 
% Regression models were built separately (Training Module) and
% therefore regression coefficients are available. Also, forecasted values
% for Chiller cooling output were estimated from building load predictions. 
% This code is meant to be used for 24 hours ahead predictions.
% The code creates an excel file and writes 
% the results on it along with time stamps

% ******** Reading data from dataset ************
Tcho = 42;%Chilled water temperature setpoint; outlet from chiller
Tcdi = 75;% Condenser water temperature; inlet temperature to chiller from tower in F
% Note that this fixed value of 75F is a placeholder.  We will ultimately
% need a means of forecasting the condenser water inlet temperature.

Qch_kW = 656.09;%building cooling load ASSIGNED TO THIS CHILLER in kW

% ***********************************************

% ********* Gordon-Ng model coefficients ********
a0 = xlsread('CH-Screw-Model-Coefficients','a1:a1');
a1 = xlsread('CH-Screw-Model-Coefficients','a2:a2');
a2 = xlsread('CH-Screw-Model-Coefficients','a3:a3');
a3 = xlsread('CH-Screw-Model-Coefficients','a4:a4');
% ***********************************************




Tcho_K= (Tcho-32)/1.8+273.15;%Converting F to Kelvin
Tcdi_K= (Tcdi-32)/1.8+273.15;%Converting F to Kelvin

COP=((Tcho_K/Tcdi_K)-a3*(Qch_kW/Tcdi_K))/((a0+(a1*(Tcho_K/Qch_kW))+a2*((Tcdi_K-Tcho_K)/(Tcdi_K*Qch_kW))+1)-((Tcho_K/Tcdi_K)-a3*(Qch_kW/Tcdi_K)));
P_Ch_In=Qch_kW/COP; %Chiller Electric Power Input in kW


