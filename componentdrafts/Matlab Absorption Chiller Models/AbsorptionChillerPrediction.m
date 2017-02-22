clc
clear all
%*************     Deployment Module       ********
% *************    Absorption Chiller    *********
 
% Regression models were built separately (Training Module) and
% therefore regression coefficients are available. Heat input to the chiller generator
% is assumed to be known and this model predicts the chiller cooling output.
% This code is meant to be used for 4 hours ahead predictions.
% The code creates an excel file and writes 
% the results on it along with time stamps.

% Dynamic Inputs
Tcho = 45.8; %Chilled water temperature setpoint; outlet from absorption chiller
Tcdi = 83.7; % Condenser water temperature; inlet temperature to absorption chiller from heat rejection in F
Tgeni = 335; % Generator inlet temperature (hot water temperature inlet to abs chiller) in F
Qin = 8.68;% heat input to the generator in mmBTU/hr

% ***********************************************

% ********* Gordon-Ng model coefficients ********
a0 = xlsread('CH-Abs-Model-Coefficients','a1:a1');
a1 = xlsread('CH-Abs-Model-Coefficients','a2:a2');
% ***********************************************




Tcho= (Tcho-32)/1.8+273.15;%Converting F to Kelvin
Tcdi= (Tcdi-32)/1.8+273.15;%Converting F to Kelvin
Tgeni= (Tgeni-32)/1.8+273.15;%Converting F to Kelvin
Qin= 293.1*Qin;%Converting mmBTU/hr to kW
    
Qch=(Qin*((Tgeni-Tcdi)/Tgeni)-a0-a1*(Tcdi/Tgeni))/((Tgeni-Tcho)/Tcho); 
Qch= Qch/3.517;%Converting kW to cooling ton

    


