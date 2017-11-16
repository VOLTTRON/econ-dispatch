clc
clear all
% *********** scenario 3- DC Power, ambient temperature, and I are measured at-site ******** 
% forecasted values for I and Ta are assumed to be available. This code is meant to be used
% for 4 hours ahead predictions- The code creates an excel file and write 
% the results (DC Power) on it along with date and time stamps

% ******** Reading data from dataset ************
If = xlsread('Dynamic-Inputs-4','c2:c5');% Forecasted I (f denotes forecasted)
Taf = xlsread('Dynamic-Inputs-4','d2:d5');% Forecasted ambient temperature in degree C
[num] = xlsread('Dynamic-Inputs-4','b2:b5');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Dynamic-Inputs-4',1, 'a2:a5');% Dates as an array of strings

% ******** PV Mount info *******
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

% ********* Diffuse model coefficients- from TMY3 data (Annual Model) ********
a1 = xlsread('Model-Coefficients','a3:a3');
a2 = xlsread('Model-Coefficients','b3:b3');
a3 = xlsread('Model-Coefficients','c3:c3');
a4 = xlsread('Model-Coefficients','d3:d3');
a5 = xlsread('Model-Coefficients','e3:e3');

% ********* G&R model coefficients (Annual Model) ********
b1 = xlsread('Model-Coefficients','g3:g3');
b2 = xlsread('Model-Coefficients','h3:h3');
b3 = xlsread('Model-Coefficients','i3:i3');
%*********************************************************

Pdc = zeros(4,1); % PV power output
n = zeros(4,1); % Day number
tstd = zeros(4,1); %Standard time
tsol = zeros(4,1);% Solar Time
month = zeros(4,1);
day = zeros(4,1);
year = zeros(4,1);
I0 = zeros(4,1);
I0norm = zeros(4,1);
Idiff = zeros(4,1);% diffuse radiation
delta = zeros(4,1); % Solar declination
omega = zeros(4,1); % Hour angle
costhetas = zeros(4,1);
Kt = zeros(4,1);
thetas = zeros(4,1); % Zenith angle of sun
sinphis = zeros(4,1);
phis = zeros(4,1); % Azimuth of sun
costhetai = zeros(4,1);
ITf = zeros(4,1);% future IT (POA irradiance)
Ibeam = zeros(4,1);% beam radiation
Ketta = zeros(4,1); % Incidence angle modifier
omegaS = zeros(4,1); % sunrise and sunset hour angle
cosomegaS = zeros(4,1); % cos of omegaS
daylightindicator = zeros(4,1); % indicates whether sun is above the horizon or not
omegaSprime = zeros(4,1); % sunset and sunrise hour angle over the panel surface
cosomegaSprime = zeros(4,1); % cos of omegaSprime
daylightindicatorT = zeros(4,1);% indicates whether sun is above the edge of the panel or not (sunrise and sunset over the panel surface)
DI = zeros(4,1);%either 0 or 1 based on daylightindicatorT

for a=1:1:4
    xxx=char(raw(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    y = char(c(3));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
    year(a)=str2num(y);%converting 'y' to numerical value
    
    %*********** calculatin n (day number) **************
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


for a=1:1:4
    %***************** Calculating I0 radiation values ************
    I0norm(a)=(1+(0.033*cos((pi/180)*(n(a)*360)/365.25)))*1367;
    tstd(a)= (num(a)*24)-0.5;%hour ending data collection
    delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(n(a)+10)/365.25);
    
    %*************** Solar Time ***************
    B=360*(n(a)-81)/364;
    B=B*(pi/180);
    Et=9.87*sin(2*B)-7.53*cos(B)-1.5*sin(B); %equation of time
    tsol(a)=tstd(a)-((Lstd-Lloc)/15)+(Et/60); %solar time in hr
    
    %****************** Calculating Kt for Idiff model ***********
    omega(a)=(tsol(a)-12)*15;
    omega(a)=omega(a)*(pi/180);
    costhetas(a)= cos(lambda)*cos(delta(a))*cos(omega(a))+sin(lambda)*sin(delta(a));%thetas is zenith angle of sun
    costhetas(a)=abs(costhetas(a));
    I0(a)=I0norm(a)*costhetas(a);

    Kt(a)=If(a)/I0(a);
    
    %*********** Calculating Idiff and Ibeam future values ****************
    Idiff(a) = If(a)*(a1+(a2*Kt(a))+(a3*Kt(a)^2)+(a4*Kt(a)^3)+(a5*Kt(a)^4));
    Ibeam(a)=If(a)-Idiff(a);
    %**********************************************************************
    
    thetas(a)=acos(costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(delta(a))*sin(omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
            
    %********** HDKR Anistropic Sky Model **********
    ITf(a)=Ibeam(a)*(1+(Idiff(a)/I0(a)))*(costhetai(a)/cos(thetas(a)))+(Idiff(a)*(1-(Ibeam(a)/I0(a)))*((1+cos(thetap))/2)*(1+(sqrt(Ibeam(a)/If(a))*sin(thetap/2)^3)))+(PVMoCo*If(a)*Rog*((1-cos(thetap))/2));
    if If(a)==0;
        ITf(a)=0;
    end
    %***********************************************
   
    %***** sunrise and sunset solar angle for horizontal surfaces ******
    cosomegaS(a)=-tan(lambda)*tan(delta(a)); % sunrise and sunset hour angle
    omegaS(a)=acos(cosomegaS(a));
    daylightindicator(a)=omegaS(a)-abs(omega(a));% negative values indicate hours before sunrise or after sunset
    cosomegaSprime(a)=-tan(lambda-thetap)*tan(delta(a));%sunrise and sunset solar angle for tilted surface with zero zenith angle
    omegaSprime(a)=acos(cosomegaSprime(a));
    daylightindicatorT(a)=omegaSprime(a)-abs(omega(a));% negative values indicate hours before sunrise or after sunset over the panel surface
    DI(a) = 1+(floor(daylightindicatorT(a)/1000));%this will creat 0 for hours before sunrise or after sunset over the panel surface and 1 for other hours
    %**************************************
    thetas(a)=acos(costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(delta(a))*sin(omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
    Ketta(a) = 1-0.1*((1/costhetai(a))-1); % incidence angle modifier
         
    %******** Power Calculation- G&R model **************
    Pdc(a)= DI(a)*(b1*ITf(a)*Ketta(a)+b2*ITf(a)*Ketta(a)*Taf(a)+b3*(ITf(a)*Ketta(a))^2);
    %*****************************************************
    
end
   %*************** Exporting Results ***************
   headers = {'Date','Time (hr)','Power Output(W)'};
   xlswrite('Power-Output-4',headers);
   xlswrite('Power-Output-4',raw,'a2:a5');
   xlswrite('Power-Output-4',num*24,'b2:b5');
   xlswrite('Power-Output-4',Pdc,'c2:c5');
   % ************************************************
