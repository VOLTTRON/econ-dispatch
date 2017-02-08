clc
clear all
% *****************     Training Module       *****************
% ***** Centrifugal Chiller (with variable speed drive control) ******

% This module reads the historical data on temperatures (in Fahrenheit), inlet power to the
% chiller (in kW) and outlet cooling load (in cooling ton); then, converts
% the data to proper units which then will be used for model training. At
% the end, regression coefficients will be written to an excel file




% ******** Reading data from dataset ************
Tcho = xlsread('CH-Cent-VSD-Historical-Data','a2:a99999');% chilled water supply temperature in F
Tcdi = xlsread('CH-Cent-VSD-Historical-Data','b2:b99999');% condenser water temperature (outlet from heat rejection and inlet to chiller) in F
Qch = xlsread('CH-Cent-VSD-Historical-Data','c2:c99999');% chiller cooling output in Tons of cooling
P = xlsread('CH-Cent-VSD-Historical-Data','d2:d99999');% chiller power input in kW
[i,j]=size(Tcho);
U=ones(i,1);

NameplateAvailable= 'Yes'; %User input of maximum chiller capacity, if available
if NameplateAvailable== 'Yes'
    Qchmax_Tons= 500;  %Chiller capacity in cooling tons
    Qchmax = Qchmax_Tons*12000/3412;
else
    Qchmax = max(Qch);
end

% *********************************

COP = zeros(i,1); % Chiller COP
x1 = zeros(i,1);
x2 = zeros(i,1);
x3 = zeros(i,1);
x4 = zeros(i,1);
y = zeros(i,1);
%tstd = zeros(24,1); % time stamp

for a=1:1:i

    Tcho(a)= (Tcho(a)-32)/1.8+273.15;%Converting F to Kelvin
    Tcdi(a)= (Tcdi(a)-32)/1.8+273.15;%Converting F to Kelvin
    Qch(a)= Qch(a)*12000/3412; % Converting tons to kW
    COP(a)= Qch(a)/P(a);
    
end

for a=1:1:i

    x1(a)=Tcho(a)/Qch(a);
    x2(a)=Tcho(a)/Qchmax;
    x3(a)=(Tcdi(a)-Tcho(a))/(Tcdi(a)*Qch(a));
    x4(a)=(((1/COP(a))+1)*Qch(a))/Tcdi(a);
    y(a)=((((1/COP(a))+1)*Tcho(a))/Tcdi(a))-1;
    
end

%*******Multiple Linear Regression***********
XX=[U,x1,x2,x3,x4];%matrix of predictors 
AA=XX\y;   %Note from Nick Fernandez: Matrix division needed for Python conversion
%********************************************

%*************** Exporting Results ***************
xlswrite('CH-Cent-VSD-Model-Coefficients',AA,'a1:a5');
xlswrite('CH-Cent-VSD-Static-Inputs',Qchmax,'a1:a1');
%*************************************************
