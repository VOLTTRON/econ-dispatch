%getting the current time which determines how the prediction method should
%work
a = clock;
t2 = a(4);
t2=10
%---------------------------------------------------------------------- 
%Reading model coefficients from excel files
model1 = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Models.xlsx','b3:j14');
model2 = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Models.xlsx','b17:f28');
%Reading input variable values from Data file
t = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','a2:a13');
cc = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','b2:b13');
cc24 = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','c2:c13');
I = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','d2:d13');
I24 = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','e2:e13');
pr4 =  xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','f2:I13');
ci = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','k2:k13');
%Specifying the number of predictions that will be made for each time period
np = [2,2,4,4,4,4,4,4,4,3,2,1];
%-------------------------------------------------------------------
%Telling to model the current month. This will be used to choose model.
v = datevec(now);
month = v(2);
%selecting appropriate models
mo1 = model1(month,1:9);
mo2 = model2(month,1:5);
%------------------------------------------------
%Defining zero matrices and vectors
le = zeros(4,2);
e = zeros(4,4);
data = zeros(12,9);
pr24 = zeros(12,1);
%--------------------------------------------------
%fiiling the first column of data with 1's
data(1:12,1) = zeros(12,1)+1;
%Putting input variables into the data matrix
data(1:12,2) = t;
data(1:12,3) = cc;
data(1:12,4) = cc24;
data(1:12,5) = I24;
%-----------------------------------------
%calculating and filling in the first lag of cloud covre and radiarion 
 data(2:12,6) = cc(1:11);
 data(2:12,8) = I(1:11);
%calculating and filling the second lag of cloud cover and radiation
data(3:12,7) = cc(1:10);
data(3:12,9) = I(1:10);
%-------------------------------------------------------
%calculating 24-hours predictions
for i=1:12
    pr24(i) = max(data(i,1:5)*mo2',0);
end
%obtaining i as the index
i = t2-7;
%-----------------------
%x = data(time-7,1:9);
if t2 >= 10 && t2 <=19
   %setting the number of predictions that will be made at ach time
   k=np(i);
   %-------------------------------------------------------------------
   for j = 0:k-1
       %The following lines of code updates  the last estimate for the time period of interest
       if j == 0
          ci(i) = ci(i)+1;
          pr4(i,ci(i)) = max(data(i,1:9)*mo1',0);
       else
           x(1:7) = data(t2-7+j,1:7);
           x(8) = pr4(i+j-1,ci(i+j-1));
           x(9) = pr4(i+j-2,ci(i+j-2));
           ci(i+j) = ci(i+j)+1;
           pr4(i+j,ci(i+j)) = max(x*mo1',0);
       end
   end
elseif t2 == 8
        ci(i) = ci(i)+1;
        pr4(i,ci(i)) = max(data(i,1:5)*mo2',0);
        ci(i+1) = ci(i+1)+1;
        pr4(i+1,ci(i)) = max(data(i+1,1:5)*mo2',0);
        x(1:7) = data(i+2,1:7);
        x(8) = pr4(i+1,ci(i+1));
        x(9) = pr4(i,ci(i));
        ci(i+2) = ci(i+2)+1;
        pr4(i+2,ci(i+2))= max(x*mo1',0);
        x(1:7) = data(i+3,1:7);
        x(8) = pr4(i+2,ci(i+2));
        x(9)= pr4(i+1,ci(i+1));
        ci(i+3) = ci(i+3)+1;
        pr4(i+3,ci(i+3)) = max(x*mo1',0);
    elseif t2 == 9
        ci(i+1) = ci(i+1)+1;
        pr4(i+1,ci(i+1)) = max(data(i+1,1:9)*mo1',0);
        x(1:7) = data(i+2,1:7);
        x(8) = pr4(i+1,ci(i+1));
        x(9) = pr4(i,ci(i+2));
        ci(i+2) = ci(i+2)+1;
        pr4(i+2,ci(i+2)) = max(x*mo1',0);
        x(1:7) = data(i+3,1:7);
        x(8) = pr4(i+2,ci(i+2));
        x(9) = pr4(i+1,ci(i+2));
        ci(i+3) = ci(i+3)+1;
        pr4(i+3,ci(i+3)) = max(x*mo1',0); 
end
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx',pr4, 'f2:i13');
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx',pr24, 'j2:j13');
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx',ci, 'k2:k13');

    
    