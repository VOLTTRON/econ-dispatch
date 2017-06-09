newdata = zeros(12,10);
newdata(1:12,2) = xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','b2:b13');
newdata(1:12,4)= xlsread('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx','d2:d13');
xlswrite('\\pnl\projects\CHPGMLC\Component Models\Task 2 Solar Forecast- Software Modules\Task 2 Solar Forecast- Software Modules\Deployment_Data.xlsx',newdata, 'b2:k13');

