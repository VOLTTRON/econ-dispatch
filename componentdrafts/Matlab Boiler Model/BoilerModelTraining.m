clc
clear all

% *********     Training Module     *******
%   *********    Boiler Model    ********

% This module reads the historical data on boiler heat output and 
% gas heat input both in mmBTU/hr; then, converts
% the data to proper units which then will be used for model training. At
% the end, regression coefficients will be written to a file

% ******** Reading data from dataset ************
Gbp = xlsread('Boiler-Historical-Data','a2:a999999');% boiler gas input in mmBTU
% Note from Nick Fernandez: Most sites will not have metering for gas inlet
% to the boiler.  I'm creating a second option to use a defualt boiler
% curve
Qbp = xlsread('Boiler-Historical-Data','b2:b999999');% boiler heat output in mmBTU
[i,j]=size(Gbp);
U=ones(i,1);

% ****** Static Inputs (Rating Condition + Natural Gas Heat Content *******
Qbprated = 60; %boiler heat output at rated condition - user input (mmBtu)
Gbprated = 90; %boiler gas heat input at rated condition - user input (mmBtu)
%**************************************************************************

xbp = zeros(i,1);
xbp2 = zeros(i,1);
xbp3 = zeros(i,1);
xbp4 = zeros(i,1);
xbp5 = zeros(i,1);
ybp = zeros(i,1);

for a=1:1:i
    
    xbp(a)=Qbp(a)/Qbprated;
    xbp2(a)= xbp(a)^2;
    xbp3(a)= xbp(a)^3;
    xbp4(a)= xbp(a)^4;
    xbp5(a)= xbp(a)^5;
    ybp(a)= (Qbp(a)/Gbp(a))/(Qbprated/Gbprated);

end

%*******Multiple Linear Regression***********
XX=[U,xbp,xbp2,xbp3,xbp4,xbp5];%matrix of predictors 
AA=XX\ybp;
%********************************************

%*************** Exporting Results ***************
xlswrite('Boiler-Model-Coefficients',AA,'a1:a6');
%*************************************************

