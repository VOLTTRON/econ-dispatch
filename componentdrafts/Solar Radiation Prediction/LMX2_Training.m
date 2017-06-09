%Reading radiation, cloud cover and time data from TMY3 excel file. Note
%that TMY3 files do not contain Month variable
Ttmp = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\phoenix_TMY3.xlsx','b3:b8762')*24;
Itmp =  xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\phoenix_TMY3.xlsx','e3:e8762');
CCtmp_x = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\phoenix_TMY3.xlsx','z3:z8762');
nmeans = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Training_Data.xlsx','a2:a2');
means_x = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Training_Data.xlsx','b2:b20');
means_xx = means_x(1:nmeans);
%----------------------------------------------------------------------
%Generating zero matrices. This way putting values in these matrices would
%be faster in subsequent parts of the code.
model1 = zeros(12,9);
model2 = zeros(12,5);
Wtmp = zeros(8760,1);
CC1tmp = zeros(8760,1);
CC2tmp = zeros(8760,1);
CC24tmp = zeros(8760,1);
I1tmp= zeros(8760,1);
I2tmp = zeros(8760,1);
I24tmp = zeros(8760,1);
Id = zeros(8736,1)+1;
a = [1,721,1393,2137,2857,3601,4321,5065,5809,6529,7273,7993];
b = [720,1392,2136,2856,3600,4320,5064,5808,6528,7272,7992,8736];
%-----------------------------------------------------------------
%calculating the daytime/night time indicator
for i = 1:8760
    if Ttmp(i)>= 8 && Ttmp(i)<= 19
        Wtmp(i) =1;
    end
end
%---------------------------------------------------------------------
%Transforming cloud cove values based on the means given
for i = 1:8760
    [c index] = min(abs(means_xx-10*CCtmp_x(i))); % Nick's Notes: I don't follow the operation going on here
    CCtmp(i) = means_xx(index); % Nick's Notes: Don't follow this either
end
%-------------------------------------------------------------------------
%calculating the first lag of cloud cover and radiation
CC1tmp(2:8760)= CCtmp(1:8759);
I1tmp(2:8760)= Itmp(1:8759);
%-------------------------------------------------------------------
%calculating the second lag of cloud cover and radiation
CC2tmp(3:8760)= CCtmp(1:8758);
I2tmp(3:8760)= Itmp(1:8758);
%-------------------------------------------------------------------
%calculating the seasonal lag of cloud cover and radiation
CC24tmp(25:8760)= CCtmp(1:8736);
I24tmp(25:8760) = Itmp(1:8736);
%------------------------------------------------------------------        
%Removing the first 24 rows as they do not contain full lag values
I = Itmp(25:8760);
CC = CCtmp(25:8760);
I1 = I1tmp(25:8760);
I2 = I2tmp(25:8760);
I24 = I24tmp(25:8760);
CC1 = CC1tmp(25:8760);
CC2 = CC2tmp(25:8760);
CC24 = CC24tmp(25:8760);
W = Wtmp(25:8760);
T = Ttmp(25:8760);
%------------------------------------------------------------------------
%forming trainign sets
x1(1:8736,1) = Id;
x1(1:8736,2) = T;
x1(1:8736,3) = CC;
x1(1:8736,4) = CC24;
x1(1:8736,5) = I24;
x1(1:8736,6) = CC1;
x1(1:8736,7) = CC2;
x1(1:8736,8) = I1;
x1(1:8736,9) = I2;
x2(1:8736,1) = Id;
x2(1:8736,2) = T;
x2(1:8736,3) = CC;
x2(1:8736,4) = CC24;
x2(1:8736,5) = I24;
%fitting The models and finding the coefficients
for i = 1:12
    trdata1 = x1(a(i):b(i),1:9);
    trdata2 = x2(a(i):b(i),1:5);
    model1(i,1:9) = lscov(trdata1,I(a(i):b(i)),W(a(i):b(i)));
    model2(i,1:5) = lscov(trdata2,I(a(i):b(i)),W(a(i):b(i)));
end
%--------------------
%Copying the results into the excel file
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Models.xlsx',model1, 'b3:j14');
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Models.xlsx',model2,'b17:f28');
%------------------------------------------------------------------------
