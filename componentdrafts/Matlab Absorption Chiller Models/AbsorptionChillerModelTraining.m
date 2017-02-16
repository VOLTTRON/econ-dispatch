clc
clear all
%*************     Training Module       ********
% *************   Absorption Chiller    *********
% This module reads the historical data on temperatures (in Fahrenheit), inlet heat to the
% chiller (in mmBTU/hr) and outlet cooling load (in cooling ton); then, converts
% the data to proper units which then will be used for model training. At
% the end, regression coefficients will be written to a file

% ******** Reading data from dataset ************
Tcho = xlsread('CH-Abs-Historical-Data','a2:a99999');% chilled water supply temperature in F
Tcdi = xlsread('CH-Abs-Historical-Data','b2:b99999');% inlet temperature from condenser in F
Tgeni = xlsread('CH-Abs-Historical-Data','c2:c99999');% generator inlet water temperature in F
Qch = xlsread('CH-Abs-Historical-Data','d2:d99999');% chiller cooling output in cooling Tons
Qin = xlsread('CH-Abs-Historical-Data','e2:e99999');% chiller heat input in mmBTU/hr
[i,j]=size(Tcho);
U=ones(i,1);

% *********************************

COP = zeros(i,1); % Chiller COP
x1 = zeros(i,1);
y = zeros(i,1);
%tstd = zeros(24,1); % time stamp

for a=1:1:i

    Tcho(a)= (Tcho(a)-32)/1.8+273.15;%Converting F to Kelvin
    Tcdi(a)= (Tcdi(a)-32)/1.8+273.15;%Converting F to Kelvin
    Tgeni(a)= (Tgeni(a)-32)/1.8+273.15;%Converting F to Kelvin
    Qch(a)= 3.517*Qch(a);%Converting cooling tons to kW
    Qin(a)= 293.1*Qin(a);%Converting mmBTU/hr to kW
    COP(a)= Qch(a)/Qin(a);
    
end

for a=1:1:i

    x1(a)=Tcdi(a)/Tgeni(a);
    y(a)=((Tgeni(a)-Tcdi(a))/(Tgeni(a)*COP(a))-((Tgeni(a)-Tcho(a))/Tcho(a)))*Qch(a);
    
end

%*******Multiple Linear Regression***********
XX=[U,x1];%matrix of predictors 
AA=XX\y;
%********************************************

%*************** Exporting Results ***************
xlswrite('CH-Abs-Model-Coefficients',AA,'a1:a2');
%*************************************************
