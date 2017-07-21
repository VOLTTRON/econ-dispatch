close all

data = xlsread('CH-Abs-Historical-Data.xlsx');

Tcho = data(:,1);% chilled water supplied temperature in F
Tcdi = data(:,2);% inlet temperature from condenser in F
Tgeni = data(:,3);% generator inlet water temperature in F
Qch = data(:,4)*3.517/293.1;% chiller cooling output in mmBTU/hr (converted from cooling Tons)
Qin = data(:,5);% chiller heat input in mmBTU/hr

Qch1 = [ones(size(Qch)) Qch];

m_AbsChiller = regress(Qin, Qch1);
xmax_AbsChiller = max(Qch);
xmin_AbsChiller = min(Qch);
x = linspace(xmin_AbsChiller, xmax_AbsChiller, 100);
y = m_AbsChiller(1)+m_AbsChiller(2)*x;

figure
scatter(Qch, Qin)
xlabel('Cooling Output Qch (mmBTU/hr)')
ylabel('Heat Input Qin (mmBTU/hr)')
hold on
plot(x,y,'r','LineWidth',3)
hold off

save('AbsChillerPara.mat','xmin_AbsChiller','xmax_AbsChiller','m_AbsChiller');