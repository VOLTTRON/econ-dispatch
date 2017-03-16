clc
clear all
%*************     Training Module       ********
% ***** Scenario 5 Power Model Training ******

% This module reads the historical data on ambient temperatures (in C), PV power 
% generation (in W) and POA irradiation (in W/m2); then, calculates the
% ketta using time stamps of the historical data and fit the power prediction model. 
% At the end, regression coefficients will be written to a file.

% ******** Reading data from dataset ************
P = xlsread('Historical Data','c2:c99999');% PV power generation in Watts
IT = xlsread('Historical Data','d2:d99999');% POA irradiation in W/m2
Ta = xlsread('Historical Data','e2:e99999');% ambient temperature in C
[num] = xlsread('Historical Data','b2:b99999');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Historical Data',1,'a2:a99999');% Dates as an array of strings
[i,j]=size(Ta);

% ******** PV Mount *******
thetap = xlsread('Static-Inputs','b3:b3'); %Plane tilt angle
thetap = thetap*(pi/180);
phip = xlsread('Static-Inputs','c3:c3'); %Plane azimuth angle
phip = phip*(pi/180);

% ******** Location Info ************
lambda = xlsread('Static-Inputs','g3:g3'); %Location latitude
Lloc = xlsread('Static-Inputs','h3:h3'); %Local longitude
Lstd = xlsread('Static-Inputs','i3:i3'); %Time zone longitude
lambda = lambda*(pi/180);

% *********************************
n = zeros(i,1); % Day number
thetas = zeros(i,1); % Zenith angle of sun
costhetas = zeros(i,1); % cos of zenith angle
costhetai = zeros(i,1); % cos of incidence angle
phis = zeros(i,1); % Azimuth of sun
sinphis = zeros(i,1); % sin of sun azimuth angle
delta = zeros(i,1); % Solar declination
tstd = zeros(i,1); %Standard time
tsol = zeros(i,1);% Solar Time
omega = zeros(i,1); % Hour angle
Ketta = zeros(i,1); % Incidence angle modifier 
month = zeros(i,1);%Month
day = zeros(i,1);%Day

x1 = zeros(i,1);
x2 = zeros(i,1);
x3 = zeros(i,1);
y = zeros(i,1);

for a=1:1:i
    xxx=char(raw(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
    %calculatin n (day number)
    if month(a)==1
        n(a)=day(a);
    elseif month(a)==2
        n(a)=31+day(a);
    elseif month(a)==3
        n(a)=59+day(a);
    elseif month(a)==4
        n(a)=90+day(a);
    elseif month(a)==5
        n(a)=120+day(a);
    elseif month(a)==6
        n(a)=151+day(a);
    elseif month(a)==7
        n(a)=181+day(a);
    elseif month(a)==8
        n(a)=212+day(a);
    elseif month(a)==9
        n(a)=243+day(a);
    elseif month(a)==10
        n(a)=273+day(a);
    elseif month(a)==11
        n(a)=304+day(a);
    elseif month(a)==12
        n(a)=334+day(a);
    end
end

for a=1:1:i
    if num(a)==0
        num(a)=1;% We want 12AM to be 24 (not 0)
    end
    tstd(a)= (num(a)*24)-0.5;%hour ending data collection
    delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(n(a)+10)/365.25);
    
    %********** Solar Time **********
    B=360*(n(a)-81)/364;
    B=B*(pi/180);
    Et=9.87*sin(2*B)-7.53*cos(B)-1.5*sin(B); %equation of time
    tsol(a)=tstd(a)-((Lstd-Lloc)/15)+(Et/60); %solar time in hr
    
    %************** Hour Angle *************
    omega(a)=(tsol(a)-12)*15;
    omega(a)=omega(a)*(pi/180);
    costhetas(a)= cos(lambda)*cos(delta(a))*cos(omega(a))+sin(lambda)*sin(delta(a));%thetas is zenith angle of sun
    costhetas(a)=abs(costhetas(a));
    %***************************************
    
    thetas(a)=acos(costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(delta(a))*sin(omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
    Ketta(a) = 1-0.1*((1/costhetai(a))-1); % incidence angle modifier
    

    %*********** Gordon and Reddy Model **********
    x1(a)=IT(a)*Ketta(a);
    x2(a)=IT(a)*Ketta(a)*Ta(a);
    x3(a)=(IT(a)*Ketta(a))^2;
    y(a)=P(a);
end

%*******Multiple Linear Regression (no intercept) ***********
XX=[x1,x2,x3];%matrix of predictors 
BB=XX\y;
%************************************************************

%*************** Exporting Results ***************
xlswrite('Model-Coefficients',BB(1),'c3:c3');
xlswrite('Model-Coefficients',BB(2),'d3:d3');
xlswrite('Model-Coefficients',BB(3),'e3:e3');
%*************************************************
