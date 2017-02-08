clc
clear all
%*****************     Deployment Module       ***************** 
% ***** Centrifugal Chiller (with variable speed drive control) ******
 
% Regression models were built separately (Training Module) and
% therefore regression coefficients are available. Also, forecasted values
% for Chiller cooling output were estimated from building load predictions. 
% This code is meant to be used for 24 hours ahead predictions.
% The code creates an excel file and writes 
% the results on it along with time stamps
NameplateAvailable= 'Yes'; %User input of maximum chiller capacity, if available
if NameplateAvailable== 'Yes'
    Qchmax_Tons= 500;  %Chiller capacity in cooling tons
    Qchmax = Qchmax_Tons*12000/3412;
else
    Qchmax = xlsread('CH-Cent-VSD-Static-Inputs','a1:a1');% estimate chiller maximum cooling output in kW from hstorical data
end

% State Variable Inputs
Tcho = 44; %Chilled water temperature setpoint; outlet from chiller
Tcdi = 75; % Condenser water temperature; inlet temperature to chiller from condenser in F
% Note that this fixed value of 75F is a placeholder.  We will ultimately
% need a means of forecasting the condenser water inlet temperature.
Qch_kW = 1758.5; %building cooling load ASSIGNED TO THIS CHILLER in kW



Qch= Qch_kW *3412/12000; % Building Cooling Load in Tons



% ********* Gordon-Ng model coefficients ********
a0 = xlsread('CH-Cent-VSD-Model-Coefficients','a1:a1');
a1 = xlsread('CH-Cent-VSD-Model-Coefficients','a2:a2');
a2 = xlsread('CH-Cent-VSD-Model-Coefficients','a3:a3');
a3 = xlsread('CH-Cent-VSD-Model-Coefficients','a4:a4');
a4 = xlsread('CH-Cent-VSD-Model-Coefficients','a5:a5');
% ***********************************************




    Tcho_K= (Tcho-32)/1.8+273.15;%Converting F to Kelvin
    Tcdi_K= (Tcdi-32)/1.8+273.15;%Converting F to Kelvin

    
   COP=((Tcho_K/Tcdi_K)-a4*(Qch_kW/Tcdi_K))/((a0+(a1+a2*(Qch_kW/Qchmax))*(Tcho_K/Qch_kW)+a3*((Tcdi_K-Tcho_K)/(Tcdi_K*Qch_kW))+1)-((Tcho_K/Tcdi_K)-a4*(Qch_kW/Tcdi_K)));
    %Coefficient of Performance(COP) of chiller from regression
    
    P_Ch_In=Qch_kW/COP; %Chiller Electric Power Input in kW


