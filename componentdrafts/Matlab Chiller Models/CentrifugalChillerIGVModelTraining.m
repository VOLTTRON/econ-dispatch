clc
clear all
%*************     Training Module       ********
% ***** Centrifugal Chiller (with inlet guide vane control) ******

% This module reads the historical data on temperatures (in Fahrenheit), inlet power to the
% chiller (in kW) and outlet cooling load (in cooling ton); then, converts
% the data to proper units which then will be used for model training. At
% the end, regression coefficients will be written to a file

% ******** Reading data from dataset ************
Tcho = xlsread('CH-Cent-IGV-Historical-Data','a2:a99999');% % chilled water supply temperature in F
Tcdi = xlsread('CH-Cent-IGV-Historical-Data','b2:b99999');% condenser water temperature (outlet from heat rejection and inlet to chiller) in F
Qch = xlsread('CH-Cent-IGV-Historical-Data','c2:c99999');% chiller cooling output in Tons of cooling
P = xlsread('CH-Cent-IGV-Historical-Data','d2:d99999');% chiller power input in kW
[i,j]=size(Tcho);
U=ones(i,1);

% *********************************

COP = zeros(i,1); % Chiller COP
x1 = zeros(i,1);
x2 = zeros(i,1);
x3 = zeros(i,1);
y = zeros(i,1);
%tstd = zeros(24,1); % time stamp

for a=1:1:i

    Tcho(a)= (Tcho(a)-32)/1.8+273.15;%Converting F to Kelvin
    Tcdi(a)= (Tcdi(a)-32)/1.8+273.15;%Converting F to Kelvin
    COP(a)= Qch(a)/P(a);
    Qch(a)= Qch(a)*12000/3412; % Converting Tons to kW
end

for a=1:1:i

    x1(a)=Tcho(a)/Qch(a);
    x2(a)=(Tcdi(a)-Tcho(a))/(Tcdi(a)*Qch(a));
    x3(a)=(((1/COP(a))+1)*Qch(a))/Tcdi(a);
    y(a)=((((1/COP(a))+1)*Tcho(a))/Tcdi(a))-1;
    
end

%*******Multiple Linear Regression***********
XX=[U,x1,x2,x3];%matrix of predictors 
AA=XX\y; %Note from Nick Fernandez: Matrix division needed for Python conversion
%********************************************

%*************** Exporting Results ***************
xlswrite('CH-Cent-IGV-Model-Coefficients',AA,'a1:a4');
%*************************************************
