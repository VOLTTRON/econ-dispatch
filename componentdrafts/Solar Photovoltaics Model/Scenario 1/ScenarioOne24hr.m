clc
clear all
% *********** scenario 1- ambient temperature, and IT are measured at-site ******** 
% forecasted values for IT and Ta are assumed to be available. This code is meant to be used
% for 24 hours ahead predictions- The code creates an excel file and write 
% the results (DC Power) on it along with date and time stamps

% ******** Reading data from dataset ************
ITf = xlsread('Dynamic-Inputs-24','c2:c25');% Forecasted IT (f denotes forecasted)
Taf = xlsread('Dynamic-Inputs-24','d2:d25');% Forecasted ambient temperature in degree C
[num] = xlsread('Dynamic-Inputs-24','b2:b25');% Time of the day as an numbers between 0 and 1
[~, ~, raw] = xlsread('Dynamic-Inputs-24',1, 'a2:a25');% Dates as an array of strings
num(24)=1;
% ************* Static Inputs *************
NOCT = xlsread('Static-Inputs','a3:a3'); %NOCT value from manufacturer's specsheet
etaRated = xlsread('Static-Inputs','b3:b3'); %Module efficiency at rated condition
TcellRated = xlsread('Static-Inputs','c3:c3'); %Module temperature at rated condition
beta = xlsread('Static-Inputs','d3:d3'); %temperature degradation coefficient
gammaDC = xlsread('Static-Inputs','e3:e3'); %DC power derate factor
gammainitial = xlsread('Static-Inputs','f3:f3');
gammaYearly = xlsread('Static-Inputs','g3:g3'); %annual module degradation
A = xlsread('Static-Inputs','i3:i3'); %total modules surface area
[~, ~, rawi] = xlsread('Static-Inputs',1, 'j3:j3');% Dates of installation (i denotes installation)
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

Pdc = zeros(24,1); % PV power output
month = zeros(24,1);
day = zeros(24,1);
year = zeros(24,1);
CurrentDate = zeros(24,1);
gammaAge = zeros(24,1);
Tcell = zeros(24,1);

for a=1:1:24
    %************** Calculating Cell Temperature ***********
    Tcell(a) = Taf(a)+((NOCT-20)/800)*ITf(a);
    %*******************************************************
    
    xxx=char(raw(a));%converting date to string
    c = strsplit(xxx,'/');%Splitting the string to get month and day
    m = char(c(1));%month as a string
    d = char(c(2));%day as a string
    y = char(c(3));%day as a string
    month(a)=str2num(m);%converting 'm' to numerical value
    day(a)=str2num(d);%converting 'd' to numerical value
    year(a)=str2num(y);%converting 'y' to numerical value
    
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
   xlswrite('Power-Output-24',raw,'a2:a25');
   xlswrite('Power-Output-24',num*24,'b2:b25');
   xlswrite('Power-Output-24',Pdc,'c2:c25');
   % ************************************************
