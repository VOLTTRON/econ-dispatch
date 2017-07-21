clear,clc
%% load parameter
load input4

%% set number of periods 
H1=1;
H2=5;
fid = fopen('BuildingOptbinary3.lp', 'w+');
fprintf(fid, '\\* Building Optimal Scheduling  *\\ \n\n');

%% objective 
fprintf(fid, 'Minimize\n Obj:');
for h=H1:H2
    fprintf(fid, ' %+.12g E_turbinegas_%d',lambda_gas(h),h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' %+.12g E_boilergas_%d',lambda_gas(h),h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' %+.12g E_gridelec_%d',lambda_elec_fromgrid(h),h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' +0 E_turbineelec_%d',h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' +0 Q_boiler_%d',h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' +0 Q_HRUheating_%d',h);
end
fprintf(fid, '\n');

for h=H1:H2
    fprintf(fid, ' +0 Q_abs_%d',h);
end
fprintf(fid, '\n');

for i=1:Nchiller
    for h=H1:H2
        fprintf(fid, ' +0 Q_chiller_%d_%d',i,h);
    end
    fprintf(fid, '\n');
end

for i=1:Nchiller
    for h=H1:H2
        fprintf(fid, ' +0 E_chillerelec_%d_%d',i,h);
    end
    fprintf(fid, '\n');
end

%% Constraints
fprintf(fid, '\n\nSubject To\n');

fprintf(fid, '\n\\* Electric energy balance *\\\n');
for h=H1:H2
    fprintf(fid, 'ElecBalance%d: E_turbineelec_%d + E_gridelec_%d',h,h,h);
    for i=1:Nchiller
        fprintf(fid, '- E_chillerelec_%d_%d',i,h);
    end
    fprintf(fid, ' = %.12g \n',E_load(h)-E_PV(h));
end

fprintf(fid, '\n\\* Heating balance *\\\n');
for h=H1:H2
    fprintf(fid, 'HeatBalance%d: Q_boiler_%d +Q_HRUheating_%d  = %.12g \n',h,h,h,Q_loadheat(h));
end

fprintf(fid, '\n\\* Cooling balance *\\\n');
for h=H1:H2
    fprintf(fid, 'CoolBalance%d: ',h);
    if flagabs==1
        fprintf(fid, 'Q_abs_%d',h);
    end
    for i=1:Nchiller
        fprintf(fid, '+Q_chiller_%d_%d', i,h);
    end
    fprintf(fid, '= %.12g \n',Q_loadcool(h));
end

fprintf(fid, '\n\\* Generator Gas *\\\n');
for h=H1:H2
    fprintf(fid, 'TurbineGasConsume%d: E_turbinegas_%d',h,h);
    for k=1:length(a_E_turbine)
        fprintf(fid, ' - %.12g E_turbineelec_%d_%d',a_E_turbine(k),h,k);
    end
    fprintf(fid, ' %+.12g Sturbine_%d = 0 \n',-b_E_turbine,h);
end

fprintf(fid, '\n\\* Generator Heat *\\\n');
for h=H1:H2
    fprintf(fid, 'TurbineHeatGenerate%d: Q_turbine_%d ',h,h);
    for k=1:length(a_Q_turbine)
        fprintf(fid, ' - %.12g E_turbineelec_%d_%d ',a_Q_turbine(k),h,k);
    end
    fprintf(fid, ' %+.12g Sturbine_%d = 0 \n',-b_Q_turbine,h);
end

fprintf(fid, '\n\\* Micro_turbine elec *\\\n');
for h=H1:H2
    fprintf(fid, 'TurbineElecGenerate%d: E_turbineelec_%d',h,h);
    for k=1:length(a_Q_turbine)
        fprintf(fid, ' - E_turbineelec_%d_%d ',h,k);
    end
    fprintf(fid, ' %+.12g Sturbine_%d =0 \n',-xmin_Turbine(1),h);
end

% fprintf(fid, '\n\\* Eturbine range *\\\n');
for h=H1:H2
    fprintf(fid, 'Eturbinelower%d: E_turbineelec_%d  %+.12g Sturbine_%d >= 0 \n',h,h,-xmin_Turbine(1),h);
end
for h=H1:H2
    fprintf(fid, 'Eturbineupper%d: E_turbineelec_%d  %+.12g Sturbine_%d <= 0 \n',h,h,-xmax_Turbine(end),h);
end

fprintf(fid, '\n\\* Boiler *\\\n');
for h=H1:H2
    fprintf(fid, 'BoilerGasConsume%d: E_boilergas_%d',h,h);
    for k=1:length(a_boiler)
        fprintf(fid, ' - %.12g Q_boiler_%d_%d',a_boiler(k),h,k);
    end
    fprintf(fid, ' %+.12g Sboiler_%d = 0 \n',-b_boiler,h);
end

for h=H1:H2
    fprintf(fid, 'BoilerHeatGenerate%d: Q_boiler_%d',h,h);
    for k=1:length(a_boiler)
        fprintf(fid, ' - Q_boiler_%d_%d ',h,k);
    end
    fprintf(fid, ' %+.12g Sboiler_%d = 0 \n',-xmin_Boiler(1),h);
end

% fprintf(fid, '\n\\* Qboiler range *\\\n');
for h=H1:H2
    fprintf(fid, 'Qboilerlower%d: Q_boiler_%d  %+.12g Sboiler_%d >= 0 \n',h,h,-xmin_Boiler(1),h);
end
for h=H1:H2
    fprintf(fid, 'Qboilerupper%d: Q_boiler_%d  %+.12g Sboiler_%d <= 0 \n',h,h,-xmax_Boiler(end),h);
end

for i=1:Nchiller
    fprintf(fid, '\n\\* Chiller %d *\\\n', i);
    for h=H1:H2
        fprintf(fid, 'ChillerElecConsume%d_%d: E_chillerelec_%d_%d',i,h,i,h);
        for k=1:length(a_chiller{i})
            fprintf(fid, ' - %.12g Q_chiller_%d_%d_%d ',a_chiller{i}(k),i,h,k);
        end
        fprintf(fid, ' %+.12g Schiller_%d_%d = 0 \n',-b_chiller{i},i,h);
    end
    
    for h=H1:H2
        fprintf(fid, 'ChillerCoolGenerate%d_%d: Q_chiller_%d_%d',i,h,i,h);
        for k=1:length(a_chiller{i})
            fprintf(fid, ' - Q_chiller_%d_%d_%d ',i,h,k);
        end
        fprintf(fid, ' %+.12g Schiller_%d_%d = 0 \n',-xmin_Chiller{i}(1),i,h);
    end
    
    for h=H1:H2
        fprintf(fid, 'Qchillerlower%d_%d: Q_chiller_%d_%d  %+.12g Schiller_%d_%d >= 0 \n',i,h,i,h,-xmin_Chiller{i}(1),i,h);
    end
    for h=H1:H2
        fprintf(fid, 'Qchillerupper%d_%d: Q_chiller_%d_%d  %+.12g Schiller_%d_%d <= 0 \n',i,h,i,h,-xmax_Chiller{i}(end),i,h);
    end
end

if flagabs==1
    fprintf(fid, '\n\\* Abschiller *\\\n');
    for h=H1:H2
        fprintf(fid, 'AbsChillerHeatCoolConsume%d: Q_Gencooling_%d',h,h);
        for k=1:length(a_abs)
            fprintf(fid, ' - %.12g Q_abs_%d_%d ',a_abs(k),h,k);
        end
        fprintf(fid, ' %+.12g Sabs_%d = 0 \n',- b_abs,h);
    end
    
    for h=H1:H2
        fprintf(fid, 'AbsChillerHeatGenerate%d: Q_abs_%d',h,h);
        for k=1:length(a_abs)
            fprintf(fid, ' - Q_abs_%d_%d ',h,k);
        end
        fprintf(fid, ' %+.12g Sabs_%d = 0 \n', -xmin_AbsChiller(1),h);
    end
    
    for h=H1:H2
        fprintf(fid, 'Qabschillerlower%d: Q_abs_%d  %+.12g Sabs_%d >= 0 \n',h,h,-xmin_AbsChiller(1),h);
    end
    for h=H1:H2
        fprintf(fid, 'Qabschillerupper%d: Q_abs_%d  %+.12g Sabs_%d <= 0 \n',h,h,-xmax_AbsChiller(end),h);
    end
end

fprintf(fid, '\n\\* HRU *\\\n');
for h=H1:H2
    fprintf(fid, 'Wasteheat%d: Q_Genheating_%d',h,h);
    if flagabs==1
        fprintf(fid, '+ Q_Gencooling_%d',h);
    end 
     fprintf(fid, '- Q_turbine_%d = 0 \n',h);
end
for h=H1:H2
    fprintf(fid, 'HRUHeatlimit%d: Q_HRUheating_%d - %.12g Q_Genheating_%d <= 0 \n',h,h,a_hru,h);
end


%% Bounds
fprintf(fid, '\n\nBounds\n');

% generator
for h=H1:H2
    fprintf(fid, '%.12g <= E_turbinegas_%d \n',xmin_Boiler(1),h);
end

for h=H1:H2
    fprintf(fid, '0 <= Q_turbine_%d \n',h);
end


for h=H1:H2
    for k=1:length(xmax_Turbine)
        fprintf(fid, '0 <= E_turbineelec_%d_%d <= %.12g\n',h,k,xmax_Turbine(k)-xmin_Turbine(k));
    end 
end


% boiler
for h=H1:H2
    fprintf(fid, '0 <= E_boilergas_%d \n',h);
end


for h=H1:H2
    for k=1:length(xmax_Boiler)
        fprintf(fid, '0 <= Q_boiler_%d_%d <= %.12g\n',h,k,xmax_Boiler(k)-xmin_Boiler(k));
    end 
end

% chiller
for i=1:Nchiller
    for h=H1:H2
        fprintf(fid, '0 <= E_chillerelec_%d_%d \n',i,h);
    end


for h=H1:H2
    for k=1:length(xmax_Chiller{i})
        fprintf(fid, '0 <= Q_chiller_%d_%d_%d <= %.12g\n',i,h,k,xmax_Chiller{i}(k)-xmin_Chiller{i}(k));
    end 
end
end

% HRU
for h=H1:H2
    fprintf(fid, '0 <= Q_HRUheating_%d \n',h);
end

for h=H1:H2
    fprintf(fid, '0 <= Q_Genheating_%d \n',h);
end

if flagabs==1
    for h=H1:H2
        fprintf(fid, '0 <= Q_Gencooling_%d\n',h);
    end
end

% Free variables
for h=H1:H2
    fprintf(fid, 'E_gridelec_%d free\n',h);
end

% Binary variables
fprintf(fid, '\n\nBinary\n');
for h=H1:H2
    fprintf(fid, 'Sturbine_%d\n',h);
end

for h=H1:H2
    fprintf(fid, 'Sboiler_%d\n',h);
end

for i=1:Nchiller
    for h=H1:H2
        fprintf(fid, 'Schiller_%d_%d\n',i,h);
    end
end

if flagabs==1
    for h=H1:H2
        fprintf(fid, 'Sabs%d\n',h);
    end
end
fprintf(fid, '\nEnd\n');
fclose(fid);


%% call glpk windoes version
tic
system('/usr/local/bin/glpsol --cpxlp BuildingOptbinary3.lp --output BuildingOptbinary_output3.txt --write BuildingOptbinary3.txt') %debug
toc
% return

%% read solution
fid1 = fopen('BuildingOptbinary3.txt');
fid2 = fopen('BuildingOptbinary3.csv','w+');
tline = fgets(fid1);

while ischar(tline)
    readstr = tline;
    readstr=regexprep(readstr,' ',',');
    fprintf(fid2,readstr);
    tline = fgets(fid1);
end
fclose(fid1);
fclose(fid2);

outputraw=csvread('BuildingOptbinary3.csv');
zeta=outputraw(2,2)
rows=outputraw(1,1);
cols=outputraw(1,2);
outputraw(1:2+rows,:)=[];

H = H2-H1+1;
Nvar = 7+2*Nchiller; %total number of variables in the objective function to plot
output = reshape(outputraw(1:Nvar*H,1),H,Nvar);

E_turbinegas = output(:,1);
E_boilergas = output(:,2);
E_gridelec = output(:,3);
E_turbineelec = output(:,4);
Q_boiler = output(:,5);
Q_HRUheating = output(:,6);
Q_abs = output(:,7);
Q_chillers = output(:,8:(7+Nchiller));
E_chillers = output(:,(8+Nchiller):end);
E_chiller_total = sum(E_chillers,2);

%% generate plots
figure(1)
subplot(2,1,1)
hold off
yyaxis left
plot(E_turbinegas.*lambda_gas(H1:H2),'*')
hold on
plot(E_boilergas.*lambda_gas(H1:H2),'o')
ylabel('Gas cost($)')
xlabel('Time (hour)')
xlim([H1-1,H2+1])
yyaxis right
plot(E_gridelec.*lambda_elec_fromgrid(H1:H2),'+')
ylabel('Electricity cost ($)')
% ylim([0,1])
legend('Generator gas cost','Boiler gas cost', 'E\_grid')
title(['Total cost for hours ', num2str(H1), ' - ', num2str(H2),': $', num2str(zeta)])

subplot(2,1,2)
hold off
yyaxis left
plot(lambda_gas)
ylabel('Gas price ($/mmBtu)')
xlabel('Time (hour)')
xlim([H1-1,H2+1])
ylim([6,8.5])
yyaxis right
plot(lambda_elec_fromgrid)
hold on 
plot(lambda_elec_togrid,'--')
ylabel('Electricity price ($/kWh)')
ylim([0,2])
legend('Gas price','Electricity buying price', 'Electricity selling price')
title('Price Information')

figure(2)
hold off
plot(Q_loadheat,'k','linewidth',2)
hold on 
plot(Q_HRUheating,'r*')
plot(Q_boiler,'bo')
xlim([H1-1,H2+1])
ylim([-0.1,max(Q_loadheat)+0.1])
legend('Q\_heating', 'Q\_HRU', 'Q\_boiler')
xlabel('Time (hour)')
ylabel('Hourly energy output (mmBtu)')
title('Heating Energy')

figure(3)
hold off
if Nchiller<4
    Q_chillers(:,(Nchiller+1):4) = 0;
end
plot(Q_loadcool,'k','linewidth',2)
hold on 
plot(Q_abs,'r*')
plot(Q_chillers(:,1),'bo')
plot(Q_chillers(:,2),'cs')
plot(Q_chillers(:,3),'gx')
plot(Q_chillers(:,4),'m+')
xlim([H1-1,H2+1])
ylim([-0.1,max(Q_loadcool)+0.1])
legend('Q\_cooling', 'Q\_abs', 'Q\_chiller1', 'Q\_chiller2', 'Q\_chiller3', 'Q\_chiller4')
xlabel('Time (hour)')
ylabel('Hourly energy output (mmBtu)')
title('Cooling Energy')

figure(4)
hold off
plot(E_load-E_PV,'k','linewidth',2)
hold on 
plot(E_turbineelec,'b*')
plot(E_chiller_total,'go')
plot(E_gridelec,'m+')
xlim([H1-1,H2+1])
ylim([min(E_grid)-0.1,max(E_load-E_PV)+0.1])
legend('E\_load - E\_PV','E\_generator', 'E\_chillers', 'E\_grid')
xlabel('Time (hour)')
ylabel('Hourly energy output (kWh)')
title('Electricity Consumption')

