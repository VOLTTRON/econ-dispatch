clc
clear all
% *********** scenario 2- ambient temperature, and I are measured at-site ******** 
% forecasted values for I and Ta are assumed to be available. This code is meant to be used
% for 24 hours ahead predictions- The code creates an excel file and write 
% the results (DC Power) on it along with date and time stamps

% ******** Reading data from dataset ************
If = xlsread('Dynamic-Inputs-24','c2:c8761');% Forecasted I (f denotes forecasted)
Taf = xlsread('Dynamic-Inputs-24','d2:d8761');% Forecasted ambient temperature in degree C
[num] = xlsread('Dynamic-Inputs-24','b2:b8761');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Dynamic-Inputs-24',1, 'a2:a8761');% Dates as an array of strings
num(8760)=1;

% ******** PV Mount info *******
thetap = xlsread('Static-Inputs','l3:l3'); %Plane tilt angle
thetap = thetap*(pi/180);
phip = xlsread('Static-Inputs','m3:m3'); %Plane azimuth angle
phip = phip*(pi/180);
PVMoCo = xlsread('Static-Inputs','n3:n3'); %PV mounting code: 1 for ground-mounted and 0 for roof-mounted systems
Rog = xlsread('Static-Inputs','o3:o3');% Ground reflectivity

% ******** Location Info ************
lambda = xlsread('Static-Inputs','q3:q3'); %Location latitude
Lloc = xlsread('Static-Inputs','r3:r3'); %Local longitude
Lstd = xlsread('Static-Inputs','s3:s3'); %Time zone longitude
lambda = lambda*(pi/180);

% ********* Diffuse model coefficients (Annual Model) ********
a1 = xlsread('Model-Coefficients','a3:a3');
a2 = xlsread('Model-Coefficients','b3:b3');
a3 = xlsread('Model-Coefficients','c3:c3');
a4 = xlsread('Model-Coefficients','d3:d3');
a5 = xlsread('Model-Coefficients','e3:e3');

% ******** Static Inputs *******
NOCT = xlsread('Static-Inputs','a3:a3'); %NOCT value from manufacturer's specsheet
etaRated = xlsread('Static-Inputs','b3:b3'); %Module efficiency at rated condition
TcellRated = xlsread('Static-Inputs','c3:c3'); %Module temperature at rated condition
beta = xlsread('Static-Inputs','d3:d3'); %temperature degradation coefficient
gammaDC = xlsread('Static-Inputs','e3:e3'); %DC power derate factor
gammainitial = xlsread('Static-Inputs','f3:f3');
gammaYearly = xlsread('Static-Inputs','g3:g3'); %annual module degradation
A = xlsread('Static-Inputs','i3:i3'); %total modules surface area
[~, ~, rawi] = xlsread('Static-Inputs',1, 'j3:j3');% Dates of installation, i denotes installation
    xxxi=char(rawi);%converting date to string
    ci = strsplit(xxxi,'/');%Splitting the string to get month and day
    mi = char(ci(1));%month as a string
    di = char(ci(2));%day as a string
    yi = char(ci(3));%year as a string
    monthi=str2num(mi);%converting 'mi' to numerical value
    dayi=str2num(di);%converting 'di' to numerical value
    yeari=str2num(yi);%converting 'yi' to numerical value
    InstallationDate = (yeari*365)+(monthi*30)+dayi;
% ********************************************************

Pdc = zeros(8760,1); % PV power output
n = zeros(8760,1); % Day number
tstd = zeros(8760,1); %Standard time
month = zeros(8760,1);
day = zeros(8760,1);
year = zeros(8760,1);
CurrentDate = zeros(8760,1);
gammaAge = zeros(8760,1);
Tcell = zeros(8760,1);
I0 = zeros(8760,1);
I0norm = zeros(8760,1);
Idiff = zeros(8760,1);
delta = zeros(8760,1);
tsol = zeros(8760,1);
omega = zeros(8760,1);
costhetas = zeros(8760,1);
Kt = zeros(8760,1);
thetas = zeros(8760,1);
sinphis = zeros(8760,1);
phis = zeros(8760,1);
costhetai = zeros(8760,1);
ITf = zeros(8760,1);
Ibeam = zeros(8760,1);

for a=1:1:8760
    xxx=char(raw(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    y = char(c(3));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
    year(a)=str2num(y);%converting 'y' to numerical value
    
    %*********** calculatin n (day number) ***********
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


for a=1:1:8760
    %***************** Calculating I0 radiation values ************
    I0norm(a)=(1+(0.033*cos((pi/180)*(n(a)*360)/365.25)))*1367;
    tstd(a)= (num(a)*24)-0.5;%hour ending data collection
    delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(n(a)+10)/365.25); %delta is Solar Declination from Cooper equation, this will be in radian
    
    %********** Solar Time **********
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
    
    %****************** Calculating Idiff and Ibeam future values *******************
    Idiff(a) = If(a)*(a1+(a2*Kt(a))+(a3*Kt(a)^2)+(a4*Kt(a)^3)+(a5*Kt(a)^4));
    Ibeam(a)=If(a)-Idiff(a);
    
    thetas(a)=acos(costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(delta(a))*sin(omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
            
   %********** HDKR Anistropic Sky Model **********
    ITf(a)=Ibeam(a)*(1+(Idiff(a)/I0(a)))*(costhetai(a)/cos(thetas(a)))+(Idiff(a)*(1-(Ibeam(a)/I0(a)))*((1+cos(thetap))/2)*(1+(sqrt(Ibeam(a)/If(a))*sin(thetap/2)^3)))+(PVMoCo*If(a)*Rog*((1-cos(thetap))/2));
    if If(a)==0;
        ITf(a)=0;
    end
   %**********************************************

   %************** Calculating Cell Temperature ***********
    Tcell(a) = Taf(a)+((NOCT-20)/800)*ITf(a);
   %*******************************************************
     
   %************** Calculating gamma Age ********************
    CurrentDate(a)=(year(a)*365)+(month(a)*30)+day(a);
    gammaAge(a) = gammainitial*(1-gammaYearly*((CurrentDate(a)-InstallationDate)/365));
    
   %*************** Calculating Power ***********************
    Pdc(a)=A*ITf(a)*etaRated*abs(1+beta*(Tcell(a)-TcellRated))*gammaDC*gammaAge(a);
   %*********************************************************
    
end
   %*************** Exporting Results ***************
   headers = {'Date','Time (hr)','Power Output(W)'};
   xlswrite('Power-Output-24',headers);
   xlswrite('Power-Output-24',raw,'a2:a8761');
   xlswrite('Power-Output-24',num*24,'b2:b8761');
   xlswrite('Power-Output-24',Pdc,'c2:c8761');
   % ************************************************
