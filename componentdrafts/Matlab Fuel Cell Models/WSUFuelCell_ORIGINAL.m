%%used to import from data file with PAFC:
% Gen.Power = DGGeneratorOutput;
% Gen.Time = linspace(1,length(Gen.Power),length(Gen.Power))';
% Gen.Fuel = DGGasInput*.293071;
% Gen.AmbTemperature = AmbientTemperature;
% Gen.AmbTemperature = 5/9*(Gen.AmbTemperature-32);
% Gen.HeatUsed = UsefulHeatRecovery*.293071;
% Gen.HeatUnused = UnusedHeatRecovery*.293071;
% Gen.HeatRecovered = Gen.HeatUsed+Gen.HeatUnused;
% Gen.Eff = Gen.Power./Gen.Fuel;
% 
% SortPower = sort(Gen.Power);
% Gen.NominalPower = round(SortPower(round(.98*length(SortPower))));
% Gen.Valid = true(length(Gen.Power),1);
% Gen.Valid(Gen.Power/Gen.NominalPower<.05) = 0;
% Gen.Valid(Gen.Power/Gen.NominalPower>1.05) = 0;
% Gen.Valid(isnan(Gen.Eff)) = 0;
% Gen.Valid(Gen.Eff<0) = 0;
% Gen.Valid(Gen.Eff<.25*Gen.Power/Gen.NominalPower) = 0;
% Gen.Valid(Gen.Eff>.45) = 0;
% Gen.Valid(Gen.AmbTemperature>33)=0;
% Gen.Valid(Gen.AmbTemperature<5)=0;

% Gen.ValidHeat = Gen.Valid;
% Gen.ValidHeat(Gen.HeatRecovered<=0) = 0;
% Gen.ValidHeat(Gen.HeatRecovered<=0.5*Gen.Power) = 0;
% 
% Gen.Start = zeros(length(Gen.Power),1);
% Gen.Hours = zeros(length(Gen.Power),1);
% for i = 11:1:length(Gen.Power)
%     if i<length(Gen.Power)-10 && max(Gen.Power(i-10:i-1))<=0 && Gen.Power(i)>0 && Gen.Power(i+10)>.5*Gen.NominalPower
%         Gen.Start(i) = Gen.Start(i-1)+1;
%     else Gen.Start(i) = Gen.Start(i-1);
%     end
%     if Gen.Power(i)>.1*Gen.NominalPower
%         Gen.Hours(i) = Gen.Hours(i-1)+1;
%     else Gen.Hours(i) = Gen.Hours(i-1);
%     end
% end

% 
% figure(5)
% C = polyfit(Gen.Power(Gen.Valid),Gen.Eff(Gen.Valid),2);
% Eff_test = C(1)*Gen.Power(Gen.Valid).^2 + C(2)*Gen.Power(Gen.Valid) + C(3);
% scatter(Gen.AmbTemperature(Gen.Valid),Gen.Eff(Gen.Valid)-Eff_test)
% C2 = polyfit(Gen.AmbTemperature(Gen.Valid),Gen.Eff(Gen.Valid)-Eff_test,1);

% save('Gen1','Gen')


%%
Coef.Fuel = 'CH4';
Coef.ThresholdDegradation = 8e3; %hours before which there is no linear degradation
Coef.NominalCurrent = [0.1921 0.1582 0.0261];
Coef.NominalASR = 0.5;
Coef.ReStartDegradation = 1e-3;
Coef.LinearDegradation = 4.5e-6;
Coef.ThresholdDegradation = 9e3;%hours before which there is no linear degradation
Coef.Utilization = [-.25, -.2 ,.65;];
Coef.StackHeatLoss = .1;
Coef.AncillaryPower = [.5, 4, .25];
Coef.Area = 5000; %5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
Coef.StackDeltaT = 100;
Coef.ExhaustTemperature = [0 0 0];
Coef.gain = 1.4;
    
    

NominalV = [0.775 0.8 0.814];
for i = 1:1:3
    load(strcat('Gen',num2str(i)));

    Coef.NominalPower = Gen.NominalPower;
    Coef.Cells = Gen.NominalPower; %currently set up for 1kW cells
    Coef.NominalOCV = NominalV(i);
    

    [FuelFlow,ExhaustFlow,ExhaustTemperature,NetEfficiency] = FuelCell_Operate(Gen.Power(Gen.Valid),Gen.AmbTemperature(Gen.Valid),Gen.Start(Gen.Valid),Gen.Hours(Gen.Valid),Coef);
    figure(i)
    hold off
    %%Fit Efficiency
    scatter(Gen.Power(Gen.Valid),Gen.Eff(Gen.Valid)*100)
    Ptest = Gen.NominalPower*linspace(0.02,1)';
    C = polyfit(Gen.Power(Gen.Valid),Gen.Eff(Gen.Valid)*100,2);
    Eff_Fit = C(1)*Ptest.^2 + C(2)*Ptest + C(3);
    hold on
    scatter(Gen.Power(Gen.Valid),NetEfficiency*100,'g')
    plot(Ptest,Eff_Fit,'r');
    
    Eff_test = C(1)*Gen.Power(Gen.Valid).^2 + C(2)*Gen.Power(Gen.Valid) + C(3);
    R_Squared = 1-sum((Gen.Eff(Gen.Valid)-Eff_test/100).^2)/sum((Gen.Eff(Gen.Valid)-mean(Gen.Eff(Gen.Valid))).^2)
    R_Squared2 = 1-sum((Gen.Eff(Gen.Valid)-NetEfficiency).^2)/sum((Gen.Eff(Gen.Valid)-mean(Gen.Eff(Gen.Valid))).^2)
    xlabel('Power (kW)','Fontsize',14)
    ylabel('Efficiency (%)','Fontsize',14)
    legend({'Data',strcat('model: r^2 =',num2str(R_Squared2,3)),strcat('fit: r^2 =',num2str(R_Squared,3))},'FontSize',12)
    %% Fit ambient Temperature Dependence (compare 1st value of C2 and C3
%     figure(5+2*i)
%     scatter(Gen.AmbTemperature(Gen.Valid),Gen.Eff(Gen.Valid)-Eff_test)
    C2 = polyfit(Gen.AmbTemperature(Gen.Valid),Gen.Eff(Gen.Valid)-Eff_test,1);
%     figure(6+2*i)
%     scatter(Gen.AmbTemperature(Gen.Valid),NetEfficiency-Eff_test)
    C3 = polyfit(Gen.AmbTemperature(Gen.Valid),NetEfficiency-Eff_test,1);
    
    %% Fit LongTerm Degradation
    Ptest = Gen.NominalPower*linspace(0.75,1)';
    Gen.Valid2 = Gen.Valid;
    Gen.Valid2(round(.1*length(Gen.Valid2)):end) = 0;
%     figure(12+2*i) %begining of life
%     hold off
%     scatter(Gen.Power(Gen.Valid2),Gen.Eff(Gen.Valid2));    
%     hold on
%     scatter(Gen.Power(Gen.Valid2),NetEfficiency(1:nnz(Gen.Valid2)),'g');
%     C = polyfit(Gen.Power(Gen.Valid2),NetEfficiency(1:nnz(Gen.Valid2)),2);
%     Eff_Fit = C(1)*Ptest.^2 + C(2)*Ptest + C(3);
%     plot(Ptest,Eff_Fit,'m');
%     C = polyfit(Gen.Power(Gen.Valid2),Gen.Eff(Gen.Valid2),2);
%     Eff_Fit = C(1)*Ptest.^2 + C(2)*Ptest + C(3);
%     plot(Ptest,Eff_Fit,'r');
    errorStart = mean(Gen.Eff(Gen.Valid2)) - mean(NetEfficiency(1:nnz(Gen.Valid2)));
    
    
    Gen.Valid2 = Gen.Valid;
    Gen.Valid2(round(1:.9*length(Gen.Valid2))) = 0;
%     figure(13+2*i) %end of life
%     hold off
%     scatter(Gen.Power(Gen.Valid2),Gen.Eff(Gen.Valid2));
%     hold on
%     scatter(Gen.Power(Gen.Valid2),NetEfficiency(end-nnz(Gen.Valid2)+1:end),'g');
%     C = polyfit(Gen.Power(Gen.Valid2),NetEfficiency(end-nnz(Gen.Valid2)+1:end),2);
%     Eff_Fit = C(1)*Ptest.^2 + C(2)*Ptest + C(3);
%     plot(Ptest,Eff_Fit,'m');
%     C = polyfit(Gen.Power(Gen.Valid2),Gen.Eff(Gen.Valid2),2);
%     Eff_Fit = C(1)*Ptest.^2 + C(2)*Ptest + C(3);
%     plot(Ptest,Eff_Fit,'r');
    errorEnd = mean(Gen.Eff(Gen.Valid2)) - mean(NetEfficiency(end-nnz(Gen.Valid2)+1:end));
end

%% default parameters
%%SOFC
% Coef.Fuel = 'CH4';
% Coef.NominalPower =100;
% Coef.NominalASR = .25;
% Coef.ReStartDegradation = 1e-4;
% Coef.LinearDegradation = 4e-6;
% Coef.ThresholdDegradation = 1e4; %hours before which there is no linear degradation
% Coef.Utilization = [-.2, -.2 ,.8];
% Coef.NominalCurrent = [0.2 0.5 0];
% Coef.StackHeatLoss = .1;
% Coef.Area = 1250; %1250cm^2 and 100 cells producing 100kW works out to 0.8 W/cm^2 and at a voltage of .8 this is 1 amp/cm^2
% Coef.Cells = 100;
% Coef.NominalOCV = 1.1;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [2, 4, 5];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.gain = 1.2;

%%PAFC
% Coef.Fuel = 'CH4';
% Coef.NominalPower =200;
% Coef.NominalASR = .5; 
% Coef.ReStartDegradation = 1e-3;
% Coef.LinearDegradation = 4.5e-6;
% Coef.ThresholdDegradation = 9e3; %hours before which there is no linear degradation
% Coef.Utilization = [-.3, -.2 ,.66]; % calibrated examples:[-.12, -.5 ,.685; -.18, -.32 ,.712;  -.11, -.30 ,.66];
% Coef.NominalCurrent = [0.1921 0.1582 0.0261];
% Coef.StackHeatLoss = .1;
% Coef.Area = 5000; %5000cm^2 and 100 cells producing 100kW works out to 0.2 W/cm^2 and at a voltage of 0.6 this is 0.333 amp/cm^2
% Coef.Cells = 200;
% Coef.NominalOCV = 0.8;
% Coef.StackDeltaT = 100;
% Coef.AncillaryPower = [.5, 4, .2];
% Coef.ExhaustTemperature = [0 0 0];
% Coef.gain = 1.4;


% [FuelFlow,ExhaustFlow,ExhaustTemperature,NetEfficiency] = FuelCell_Operate(Coef.NominalPower,18,0,0,Coef)