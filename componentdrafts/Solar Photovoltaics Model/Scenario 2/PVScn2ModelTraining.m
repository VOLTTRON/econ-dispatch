clc
clear all
%*************     Training Module       ********
% ***** Scenario 2 Power Model Training ******

% This module reads the TMY3 data on diffuse and GHI radiation (in W/m2); then, calculates the
% kt for diffuse radiation model; then, Fit diffuse radiation model and 
% at the end, regression coefficients will be written to a file.

% ******** Reading data from dataset ************
% ******* TMY file shall be in xls format **********
ITMY = xlsread('Chicago.csv','k20:k8779');% global horizontal irradiation from TMY3 file in W/m2
IdiffTMY = xlsread('Chicago.csv','m20:m8779');% diffuse radiation in W/m2
[numTMY] = xlsread('Chicago.csv','b20:b8779');% Time of the day as an numbers between 0 and 1
[~, ~, rawTMY] = xlsread('Chicago.csv',1,'a20:a8779');% Dates as an array of strings
[i,j]=size(ITMY);

% ******** Location Info ************
lambda = xlsread('Static-Inputs','q3:q3'); %Location latitude
Lloc = xlsread('Static-Inputs','r3:r3'); %Local longitude
Lstd = xlsread('Static-Inputs','s3:s3'); %Time zone longitude
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
    %***************************************
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

%*************** Exporting Results ***************
xlswrite('Model-Coefficients',AA(1),'a3:a3');
xlswrite('Model-Coefficients',AA(2),'b3:b3');
xlswrite('Model-Coefficients',AA(3),'c3:c3');
xlswrite('Model-Coefficients',AA(4),'d3:d3');
xlswrite('Model-Coefficients',AA(5),'e3:e3');
%*************************************************
