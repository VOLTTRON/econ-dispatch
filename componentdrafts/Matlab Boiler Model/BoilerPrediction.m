clc
clear all
% *********    Deployment Module    *******
%   *********    Boiler Model    ********

% here we assumed that regression models were built separately and
% therefore regression model coefficients are available. also, forecasted values
% for boiler heat output were estimated from building load predictions. 
% This code is meant to be used for 24 hours ahead predictions. 

Qbp = 55;% Building heating load assigned to Boiler

%************************************************

% ****** Boiler Nameplate parameters (User Inputs)
Qbprated = 60; %mmBtu/hr
Gbprated = 90; % mmBtu/hr
GasInputSubmetering='Yes'; %Is metering of gas input to the boilers available?  If not, we can't build a regression, and instead will rely on default boiler part load efficiency curves
%************************************************************
HC = 0.03355; %NG heat Content 950 Btu/ft3 is assumed
% **************************************************************************


if GasInputSubmetering(1) == 'Y'  
    % ********* 5-degree polynomial model coefficients from training*****
    a0 = xlsread('Boiler-Model-Coefficients','a1:a1');
    a1 = xlsread('Boiler-Model-Coefficients','a2:a2');
    a2 = xlsread('Boiler-Model-Coefficients','a3:a3');
    a3 = xlsread('Boiler-Model-Coefficients','a4:a4');
    a4 = xlsread('Boiler-Model-Coefficients','a5:a5');
    a5 = xlsread('Boiler-Model-Coefficients','a6:a6');
    % *********************************************************
else 
    % Use part load curve for 'atmospheric' boiler from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.553.4931&rep=rep1&type=pdf
    a0 = 0.6978;
    a1 = 3.3745;
    a2 = -15.632;
    a3 = 32.772;
    a4 = -31.45;
    a5 = 11.268;
end


if Qbp>Qbprated
    Qbp=Qbprated;
end
xbp=Qbp/Qbprated; % part load ratio
ybp=a0+a1*xbp+a2*(xbp)^2+a3*(xbp)^3+a4*(xbp)^4+a5*(xbp)^5;% relative efficiency (multiplier to ratred efficiency)
Gbp=(Qbp*Gbprated)/(ybp*Qbprated);% boiler gas heat input in mmBtu
FC=Gbp/HC; %fuel consumption in cubic meters per hour 

   


