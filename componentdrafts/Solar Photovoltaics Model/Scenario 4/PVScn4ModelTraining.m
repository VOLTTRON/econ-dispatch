clc
clear all
%*************     Training Module       ********
% ***** Scenario 4 Power Model Training ******

% This module reads the historical data on ambient temperatures (in C), PV power 
% generation (in W) and GHI and diffuse irradiation (in W/m2); then, fits
% the diffuse radiation model. POA irradince will be calculated usnig the HDKR model.
% Calculated POA irradiance will be used to fit the power prediction model.
% At the end, regression coefficients will be written to a file.

% ******** Reading data from dataset ************
P = xlsread('Historical Data','c2:c99999');% PV power generation in Watts
I = xlsread('Historical Data','d2:d99999');% global horizontal irradiance in W/m2
Ta = xlsread('Historical Data','e2:e99999');% ambient temperature in C
Idiff = xlsread('Historical Data','f2:f99999');% diffuse radiation in W/m2
[num] = xlsread('Historical Data','b2:b99999');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Historical Data',1,'a2:a99999');% Dates as an array of strings
[i,j]=size(Ta);

% ******** PV Mount *******
thetap = xlsread('Static-Inputs','a3:a3'); %Plane tilt angle
thetap = thetap*(pi/180);
phip = xlsread('Static-Inputs','b3:b3'); %Plane azimuth angle
phip = phip*(pi/180);
PVMoCo = xlsread('Static-Inputs','c3:c3'); %PV mounting code: 1 for ground-mounted and 0 for roof-mounted systems
Rog = xlsread('Static-Inputs','d3:d3');% Ground reflectivity

% ******** Location Info ************
lambda = xlsread('Static-Inputs','f3:f3'); %Location latitude
Lloc = xlsread('Static-Inputs','g3:g3'); %Local longitude
Lstd = xlsread('Static-Inputs','h3:h3'); %Time zone longitude
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
I0norm = zeros(i,1);
I0 = zeros(i,1);
Ibeam = zeros(i,1);
IT = zeros(i,1);
Kt = zeros(i,1);

z1 = zeros(i,1);
z2 = zeros(i,1);
z3 = zeros(i,1);
z4 = zeros(i,1);
w = zeros(i,1);

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

    %*************************************
    if num(a)==0
        num(a)=1;% We want 12AM to be 24 (not 0)
    end
    %*************************************
    
    %********** Solar Time **********
    tstd(a)= (num(a)*24)-0.5;%hour ending data collection
    B=360*(n(a)-81)/364;
    B=B*(pi/180);
    Et=9.87*sin(2*B)-7.53*cos(B)-1.5*sin(B); %equation of time
    tsol(a)=tstd(a)-((Lstd-Lloc)/15)+(Et/60); %solar time in hr
    
    %************** Hour Angle *************
    delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(n(a)+10)/365.25);
    omega(a)=(tsol(a)-12)*15;
    omega(a)=omega(a)*(pi/180);
    costhetas(a)= cos(lambda)*cos(delta(a))*cos(omega(a))+sin(lambda)*sin(delta(a));%thetas is zenith angle of sun
    costhetas(a)=abs(costhetas(a));

    %*************** Idiff model **************
    I0norm(a)=(1+(0.033*cos((pi/180)*(n(a)*360)/365.25)))*1367;
    I0(a)=I0norm(a)*costhetas(a);
    Kt(a)=I(a)/I0(a);
    z1(a)=Kt(a);
    z2(a)=(Kt(a))^2;
    z3(a)=(Kt(a))^3;
    z4(a)=(Kt(a))^4;
    if I(a)==0
        I(a)=0.00001;% Otherwise w will be NaN
    end
    w(a)=Idiff(a)/I(a);
    %******************************************
    
    thetas(a)=acos(costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(delta(a))*sin(omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
    Ketta(a) = 1-0.1*((1/costhetai(a))-1); % incidence angle modifier
    
    %******* Calculating POA Irradiation *******
    Ibeam(a)=I(a)-Idiff(a);
    %********** HDKR Anistropic Sky Model **********
    IT(a)=Ibeam(a)*(1+(Idiff(a)/I0(a)))*(costhetai(a)/cos(thetas(a)))+(Idiff(a)*(1-(Ibeam(a)/I0(a)))*((1+cos(thetap))/2)*(1+(sqrt(Ibeam(a)/I(a))*sin(thetap/2)^3)))+(PVMoCo*I(a)*Rog*((1-cos(thetap))/2));
    if I(a)==0;
        IT(a)=0;
    end

    %*********** Gordon and Reddy Model **********
    x1(a)=IT(a)*Ketta(a);
    x2(a)=IT(a)*Ketta(a)*Ta(a);
    x3(a)=(IT(a)*Ketta(a))^2;
    y(a)=P(a);
end

%******* Idiff Model: Multiple Linear Regression ***********
U=ones(i,1);
ZZ=[U,z1,z2,z3,z4];%matrix of predictors 
AA=ZZ\w;

%******* Power Model: Multiple Linear Regression (no intercept) ***********
XX=[x1,x2,x3];%matrix of predictors 
BB=XX\y;
%************************************************************

%*************** Exporting Results ***************
xlswrite('Model-Coefficients',AA(1),'a3:a3');
xlswrite('Model-Coefficients',AA(2),'b3:b3');
xlswrite('Model-Coefficients',AA(3),'c3:c3');
xlswrite('Model-Coefficients',AA(4),'d3:d3');
xlswrite('Model-Coefficients',AA(5),'e3:e3');
xlswrite('Model-Coefficients',BB(1),'g3:g3');
xlswrite('Model-Coefficients',BB(2),'h3:h3');
xlswrite('Model-Coefficients',BB(3),'i3:i3');
%*************************************************
