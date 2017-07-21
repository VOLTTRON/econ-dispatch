close all

data = xlsread('CH-Cent-IGV-Historical-Data.xlsx');

Tcho = data(:,1);% chilled water supply temperature in F
Tcdi = data(:,2);% inlet temperature from condenser in F
Qch = data(:,3)*3.517/293.1;% chiller cooling output in mmBtu/hr (converted from cooling Tons)
P = data(:,4);% chiller power input in kW

Qch1 = [ones(size(Qch)) Qch];

m_ChillerIGV = regress(P, Qch1);
xmax_ChillerIGV = max(Qch);
xmin_ChillerIGV = min(Qch);
x = linspace(xmin_ChillerIGV, xmax_ChillerIGV, 100);
y = m_ChillerIGV(1)+m_ChillerIGV(2)*x;

figure
scatter(Qch, P)
xlabel('Cooling Output Qch (mmBTU/hr)')
ylabel('Power Input P (kW)')
hold on
plot(x,y,'r','LineWidth',3)
hold off

save('ChillerIGVPara.mat','xmin_ChillerIGV','xmax_ChillerIGV','m_ChillerIGV');

% data_sort = sortrows([Qch,P]);
% P = data_sort(:,2);
% Qch = data_sort(:,1);
% N = length(P);
% n1 = find(Qch<110,1,'last');
% xmin_ChillerIGV = zeros(2,1);
% xmax_ChillerIGV = zeros(2,1);
% Qch1 = [ones(N,1), Qch];
% m1 = regress(P(1:n1), Qch1(1:n1,:));
% m2 = regress(P(n1:end), Qch1(n1:end,:));
% m_ChillerIGV = [m1,m2];
% xmax_ChillerIGV(1) = max(Qch(1:n1));
% xmin_ChillerIGV(1) = min(Qch(1:n1));
% xmax_ChillerIGV(2) = max(Qch(n1:end));
% xmin_ChillerIGV(2) = xmax_ChillerIGV(1);
% 
% figure
% scatter(Qch, P)
% xlabel('Qch (Tons)')
% ylabel('P (kW)')
% hold on
% for i=1:2
%     x = linspace(xmin_ChillerIGV(i), xmax_ChillerIGV(i), 100);
%     y = m_ChillerIGV(1,i)+m_ChillerIGV(2,i)*x;
%     plot(x,y,'r','LineWidth',3)
% end
% hold off