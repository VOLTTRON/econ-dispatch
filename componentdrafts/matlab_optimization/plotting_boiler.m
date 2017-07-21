% clear
close all

Gbp = xlsread('Boiler-Historical-Data','a:a');% boiler gas heat output in mmBTU
Qbp = xlsread('Boiler-Historical-Data','b:b');% boiler heat output in mmBTU

data_sort = sortrows([Qbp, Gbp]);
Qbp = data_sort(:,1);
Gbp = data_sort(:,2);

N = length(Qbp);
n1 = find(Qbp<24,1,'last');
n2 = find(Qbp<45,1,'last');
xmin_Boiler = zeros(3,1);
xmax_Boiler = zeros(3,1);
Qbp1 = [ones(N,1), Qbp];
m1 = regress(Gbp(1:n1), Qbp1(1:n1,:));
m2 = regress(Gbp(n1:n2), Qbp1(n1:n2,:));
m3 = regress(Gbp(n2:end), Qbp1(n2:end,:));
m_Boiler = [m1,m2,m3];
disp(m_Boiler);
xmax_Boiler(1) = max(Qbp(1:n1));
xmin_Boiler(1) = min(Qbp(1:n1));
xmax_Boiler(2) = max(Qbp(n1:n2));
xmin_Boiler(2) = xmax_Boiler(1);
xmax_Boiler(3) = max(Qbp(n2:end));
xmin_Boiler(3) = xmax_Boiler(2);

x1 = xmax_Boiler(1);
x2 = xmin_Boiler(3);
y1 = m_Boiler(1,1)+m_Boiler(2,1)*xmax_Boiler(1);
y2 = m_Boiler(1,3)+m_Boiler(2,3)*xmin_Boiler(3);

m_Boiler(2,2) = (y2-y1)/(x2-x1);
m_Boiler(1,2) = y1-m_Boiler(2,2)*x1;

figure(1)
scatter(Qbp,Gbp,'.')
xlabel('Boiler Heat Output (mmBTU)')
ylabel('Gas Heat Input (mmBTU)')
hold on
for i=1:3
    x = linspace(xmin_Boiler(i), xmax_Boiler(i), 100);
    y = m_Boiler(1,i)+m_Boiler(2,i)*x;
    plot(x,y,'r','LineWidth',3)
end
hold off

save('BoilerPara.mat','xmin_Boiler','xmax_Boiler','m_Boiler');


