clc
clear all
% *********** scenario 5- DC Power, ambient temperature, and IT are measured at-site ******** 
% here we assumed that regression models were built separately and
% therefore regression coefficients are available. also, forecasted values
% for IT and Ta are assumed to be available. This code is meant to be used
% for 24 hours ahead predictions- The code creates an excel file named
% "power-Output" and write the results (DC Power) on it along with date and
% time stamps

% ******** Reading Forecasted Solar and Ambient Temperatures ************
ITf = xlsread('Dynamic-Inputs','c2:c25');% Forecasted Total solar irradiance on tilted plane or POA irradiance, [W/m2]
Taf = xlsread('Dynamic-Inputs','d2:d25');% Forecasted ambient temperature, [Degrees C]
[TOD] = xlsread('Dynamic-Inputs','b2:b25');% Time of the day as a numbers between 0 and 1
[~, ~, raw] = xlsread('Dynamic-Inputs',1, 'a2:a25');% Dates as an array of strings

% ******** PV Mount, User inputs *******
theta_p = 15; %Tilt of PV panels with respect to horizontal, degrees
theta_p = theta_p*(pi/180); %converting degrees to radians
phi_p = 0; % Azimuth of plane. South facing is 0 (positive for orientations west of south), degrees
phi_p = phi_p*(pi/180);%converting degrees to radians
%PVMoCo = xlsread('Static-Inputs','d3'); %PV mounting code: 1 for ground-mounted and 0 for roof-mounted systems
%Rog = xlsread('Static-Inputs','e3');% Ground reflectivity

% ******** Location Info, User Input OR obtain somehow from zip code or other smart means ************
lambda = 39.74; %Location latitude
Lloc = -105.18; %Local longitude

TimeZone = 'MST';
if TimeZone == 'MST'
    Lstd=-105;
elseif TimeZone == 'PST'
    Lstd=-120;
elseif TimeZone == 'CST'
    Lstd=-90;
elseif TimeZone == 'EST'
    Lstd=-75;
else
    Lstd=-120;
end

lambda = lambda*(pi/180);%converting degrees to radians

% ********* G&R model coefficients (Annual Model) ********
a1 = xlsread('Model-Coefficients','c3:c3');
a2 = xlsread('Model-Coefficients','d3:d3');
a3 = xlsread('Model-Coefficients','e3:e3');

% ********************************************************
%Pac = zeros(24,1); % Inverter power output
Pdc = zeros(24,1); % PV power output, DC
n = zeros(24,1); % Day number
theta_s = zeros(24,1); % Zenith angle of sun
cos_theta_s = zeros(24,1); % cos of zenith angle
cos_theta_i = zeros(24,1); % cos of incidence angle
phi_s = zeros(24,1); % Azimuth of sun
sin_phi_s = zeros(24,1); % sin of sun azimuth angle
delta = zeros(24,1); % Solar declination
%sindelta = zeros(24,1); % sin of solar declination
t_std = zeros(24,1); %Standard time
t_sol = zeros(24,1);% Solar Time
omega = zeros(24,1); % Hour angle
Ketta = zeros(24,1); % Incidence angle modifier 
omegaS = zeros(24,1); % sunrise and sunset hour angle
cos_omega_S = zeros(24,1); % cos of omegaS
daylightindicator = zeros(24,1); % indicates whether sun is above the horizon or not
omegaSprime = zeros(24,1); % sunset and sunrise hour angle over the panel surface
cos_omegaS_prime = zeros(24,1); % cos of omegaSprime
daylightindicatorT = zeros(24,1);% indicates whether sun is above the edge of the panel or not (sunrise and sunset over the panel surface)
DI = zeros(24,1);%either 0 or 1 based on daylightindicatorT
month = zeros(24,1);%Month
day = zeros(24,1);%Day
%********* Calculating n (day number) **********
for a=1:1:24
    xxx=char(raw(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
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

for a=1:1:24
    t_std(a)= (TOD(a)*24)-0.5;%hour ending data collection
    delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(n(a)+10)/365.25);
    %***** Solar Time *****
    B=360*(n(a)-81)/364;
    B=B*(pi/180);
    Et=9.87*sin(2*B)-7.53*cos(B)-1.5*sin(B); %equation of time
    t_sol(a)=t_std(a)-((Lstd-Lloc)/15)+(Et/60); %solar time in hr
    %****** Hour Angle *************
    omega(a)=(t_sol(a)-12)*15;
    omega(a)=omega(a)*(pi/180);
    cos_theta_s(a)= cos(lambda)*cos(delta(a))*cos(omega(a))+sin(lambda)*sin(delta(a));%thetas is zenith angle of sun
    cos_theta_s(a)=abs(cos_theta_s(a));
    %***** sunrise and sunset solar angle for horizontal surfaces ******
    cos_omega_S(a)=-tan(lambda)*tan(delta(a)); % sunrise and sunset hour angle
    omegaS(a)=acos(cos_omega_S(a));
    daylightindicator(a)=omegaS(a)-abs(omega(a));% negative values indicate hours before sunrise or after sunset
    cos_omegaS_prime(a)=-tan(lambda-theta_p)*tan(delta(a));%sunrise and sunset solar angle for tilted surface with zero zenith angle
    omegaSprime(a)=acos(cos_omegaS_prime(a));
    daylightindicatorT(a)=omegaSprime(a)-abs(omega(a));% negative values indicate hours before sunrise or after sunset over the panel surface
    DI(a) = 1+(floor(daylightindicatorT(a)/1000));
    
    % **************************************
    theta_s(a)=acos(cos_theta_s(a)); %thetas will be in radian
    sin_phi_s(a)=(cos(delta(a))*sin(omega(a)))/sin(theta_s(a)); %phis is azimuth of sun
    phi_s(a)=asin(sin_phi_s(a));
    cos_theta_i(a)=(sin(theta_s(a))*sin(theta_p)*cos(phi_s(a)-phi_p))+(cos(theta_s(a))*cos(theta_p)); %thetai is solar incidence angle on plane
    Ketta(a) = 1-0.1*((1/cos_theta_i(a))-1); % incidence angle modifier
    
    % ******** Power Calculation- G&R model **************
    Pdc(a)= DI(a)*(a1*ITf(a)*Ketta(a)+a2*ITf(a)*Ketta(a)*Taf(a)+a3*(ITf(a)*Ketta(a))^2);
    %*****************************************************
end

