clc
clear all
%*************     Training Module       ********
% ***** Scenario 3 Power Model Training ******

% This module reads the historical data on ambient temperatures (in C), PV power 
% generation (in W) and GHI (in W/m2); also reads diffuse and GHI radiation from TMY3 dataset.
% Then, fits the diffuse radiation model and uses the model coefficients to
% calculate the Idiff for the historical dataset and then the beam
% radiation will be calculated; using these, it then calculates the POA
% irradiance for the historical dataset which along with power and ambient
% temperature data will be used for power model training.
% At the end, regression coefficients will be written to a file.

% ******** Reading data from dataset ************
% ******* TMY file shall be in xls format **********
ITMY = xlsread('726980TYA','e3:e8762');% global horizontal irradiation from TMY3 file in W/m2
IdiffTMY = xlsread('726980TYA','k3:k8762');% diffuse radiation in W/m2
[numTMY] = xlsread('726980TYA','b3:b8762');% Time of the day as an numbers between 0 and 1
[~, ~, rawTMY] = xlsread('726980TYA',1,'a3:a8762');% Dates as an array of strings
[i,j]=size(ITMY);

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
delta = zeros(i,1); % Solar declination
tstd = zeros(i,1); %Standard time
tsol = zeros(i,1);% Solar Time
omega = zeros(i,1); % Hour angle
month = zeros(i,1);%Month
day = zeros(i,1);%Day
I0norm = zeros(i,1);
I0 = zeros(i,1);
Kt = zeros(i,1);
costhetas = zeros(i,1);

z1 = zeros(i,1);
z2 = zeros(i,1);
z3 = zeros(i,1);
z4 = zeros(i,1);
w = zeros(i,1);


for a=1:1:i
    xxx=char(rawTMY(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
    %calculatin n (day number)for TMY3 dataset
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
    if numTMY(a)==0
        numTMY(a)=1;% We want 12AM to be 24 (not 0)
    end
    %*************************************
    
    %********** Solar Time **********
    tstd(a)= (numTMY(a)*24)-0.5;%hour ending data collection
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
    Kt(a)=ITMY(a)/I0(a);
    z1(a)=Kt(a);
    z2(a)=(Kt(a))^2;
    z3(a)=(Kt(a))^3;
    z4(a)=(Kt(a))^4;
    if ITMY(a)==0
        ITMY(a)=0.00001;% Otherwise w will be NaN
    end
    w(a)=IdiffTMY(a)/ITMY(a);
    %******************************************

end

%******* Idiff Model: Multiple Linear Regression ***********
U=ones(i,1);
ZZ=[U,z1,z2,z3,z4];%matrix of predictors 
AA=ZZ\w;
%***********************************************************

P = xlsread('Historical Data','c2:c99999');% PV power generation in Watts
I = xlsread('Historical Data','d2:d99999');% PV power generation in Watts
Ta = xlsread('Historical Data','e2:e99999');% ambient temperature in C
[num] = xlsread('Historical Data','b2:b99999');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Historical Data',1,'a2:a99999');% Dates as an array of strings
[u,v]=size(P);

Idiff = zeros(u,1);
MONTH = zeros(u,1);
DAY = zeros(u,1);
N = zeros(u,1);
Delta = zeros(u,1);
Tstd = zeros(u,1);
I0NORM = zeros(u,1);
Ibeam = zeros(u,1);
b = zeros(u,1);
ET = zeros(u,1);
Omega = zeros(u,1);
Tsol = zeros(u,1);
Costhetas = zeros(u,1);
I00 = zeros(u,1);
KT = zeros(u,1);
Ketta = zeros(u,1); % Incidence angle modifier
costhetai = zeros(u,1); % cos of incidence angle
IT = zeros(u,1);
thetas = zeros(u,1); % Zenith angle of sun
phis = zeros(u,1); % Azimuth of sun
sinphis = zeros(u,1); % sin of sun azimuth angle

x1 = zeros(u,1);
x2 = zeros(u,1);
x3 = zeros(u,1);
y = zeros(u,1);

for a=1:1:u
    XXX=char(raw(a));%converting date to string
    C = strsplit(XXX,'/');%Splitting the string to get month and day
    M = char(C(1));%month as a string
    D = char(C(2));%day as a string
    MONTH(a)=str2num(M);%converting 'm' to numerical value
    DAY(a)=str2num(D);%converting 'd' to numerical value
    %calculatin N (day number)for historical dataset
    if MONTH(a)==1
        N(a)=DAY(a);
    elseif MONTH(a)==2
        N(a)=31+DAY(a);
    elseif MONTH(a)==3
        N(a)=59+DAY(a);
    elseif MONTH(a)==4
        N(a)=90+DAY(a);
    elseif MONTH(a)==5
        N(a)=120+DAY(a);
    elseif MONTH(a)==6
        N(a)=151+DAY(a);
    elseif MONTH(a)==7
        N(a)=181+DAY(a);
    elseif MONTH(a)==8
        N(a)=212+DAY(a);
    elseif MONTH(a)==9
        N(a)=243+DAY(a);
    elseif MONTH(a)==10
        N(a)=273+DAY(a);
    elseif MONTH(a)==11
        N(a)=304+DAY(a);
    elseif MONTH(a)==12
        N(a)=334+DAY(a);
    end
    
    %*************************************
    if num(a)==0
        num(a)=1;% We want 12AM to be 24 (not 0)
    end
    %*************************************
    
    %********** Solar Time **********
    Tstd(a)= (num(a)*24)-0.5;%hour ending data collection
    b=360*(N(a)-81)/364;
    b=b*(pi/180);
    ET=9.87*sin(2*b)-7.53*cos(b)-1.5*sin(b); %equation of time
    Tsol(a)=Tstd(a)-((Lstd-Lloc)/15)+(ET/60); %solar time in hr
    
    %************** Hour Angle *************
    Delta(a)=-sin((pi/180)*23.45)*cos((pi/180)*360*(N(a)+10)/365.25);
    Omega(a)=(Tsol(a)-12)*15;
    Omega(a)=Omega(a)*(pi/180);
    Costhetas(a)= cos(lambda)*cos(Delta(a))*cos(Omega(a))+sin(lambda)*sin(Delta(a));%thetas is zenith angle of sun
    Costhetas(a)=abs(Costhetas(a));

    %*************** Idiff model **************
    I0NORM(a)=(1+(0.033*cos((pi/180)*(N(a)*360)/365.25)))*1367;    
    I00(a)=I0NORM(a)*Costhetas(a);
    KT(a)=I(a)/I00(a);

    %******* Calculating POA Irradiation *******
    Idiff(a)=I(a)*(AA(1)+AA(2)*KT(a)+AA(3)*KT(a)^2+AA(4)*KT(a)^3+AA(5)*KT(a)^4);
    Ibeam(a)=I(a)-Idiff(a);
    %*******************************************
    
    thetas(a)=acos(Costhetas(a)); %thetas will be in radian
    sinphis(a)=(cos(Delta(a))*sin(Omega(a)))/sin(thetas(a)); %phis is azimuth of sun
    phis(a)=asin(sinphis(a));
    costhetai(a)=(sin(thetas(a))*sin(thetap)*cos(phis(a)-phip))+(cos(thetas(a))*cos(thetap)); %thetai is solar incidence angle on plane
    
    %********** HDKR Anistropic Sky Model **********
    IT(a)=Ibeam(a)*(1+(Idiff(a)/I00(a)))*(costhetai(a)/cos(thetas(a)))+(Idiff(a)*(1-(Ibeam(a)/I00(a)))*((1+cos(thetap))/2)*(1+(sqrt(Ibeam(a)/I(a))*sin(thetap/2)^3)))+(PVMoCo*I(a)*Rog*((1-cos(thetap))/2));
    if I(a)==0;
        IT(a)=0;
    end
    
    %************ Calculating Ketta ******************
    Ketta(a) = 1-0.1*((1/costhetai(a))-1); % incidence angle modifier
    
    %*********** Gordon and Reddy Model **********
    x1(a)=IT(a)*Ketta(a);
    x2(a)=IT(a)*Ketta(a)*Ta(a);
    x3(a)=(IT(a)*Ketta(a))^2;
    y(a)=P(a);
end

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
