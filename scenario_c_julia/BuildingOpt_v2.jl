using DataFrames #to use readtable and dataframe
using Parameters
using JuMP
# using Ipopt
using Cbc

cd(dirname(@__FILE__))

parasys= readtable("para_sys.csv"); # price and loading information

@with_kw struct BuildAsset
    fundata
    ramp_up::Float64 =Inf
    ramp_down::Float64 =-Inf
    startcost::Float64 = 0
    min_on::Int = 0
    min_off::Int = 0
end


@with_kw mutable struct BuildAsset_init
    status::Int = 0
    status1::Vector{Int} = zeros(24)
    output::Float64 = 0.0
end

@with_kw struct storage
    pmax::Float64
    Emax::Float64
    eta_ch::Float64
    eta_disch::Float64
    soc_max::Float64 = 1.0
    soc_min::Float64 = 0.0
end

turbine_para = BuildAsset[];
push!(turbine_para, BuildAsset(fundata=readtable("paraturbine1.csv"),ramp_up=150,ramp_down=-150,startcost=20,min_on=3)); # fuel cell
push!(turbine_para, BuildAsset(fundata=readtable("paraturbine2.csv"),ramp_up=200,ramp_down=-200,startcost=10,min_on=3)); # microturbine
push!(turbine_para, BuildAsset(fundata=readtable("paraturbine3.csv"),ramp_up=20,ramp_down=-20,startcost=5,min_on=3)); # diesel

turbine_init=BuildAsset_init[];
push!(turbine_init,BuildAsset_init(status=1,output=300.0));
push!(turbine_init,BuildAsset_init(status=1,output=250.0));
push!(turbine_init,BuildAsset_init(status=0));
turbine_init[1].status1[22:24]=1;
turbine_init[2].status1[22:24]=1;


boiler_para = BuildAsset[];
push!(boiler_para, BuildAsset(fundata=readtable("paraboiler1.csv"),ramp_up=8,ramp_down=-8,startcost=0.8)); # boiler1
push!(boiler_para, BuildAsset(fundata=readtable("paraboiler2.csv"),ramp_up=2,ramp_down=-2,startcost=0.25)); # boiler2

boiler_init=BuildAsset_init[];
push!(boiler_init,BuildAsset_init(status=0));
push!(boiler_init,BuildAsset_init(status=0));


chiller_para = BuildAsset[];
push!(chiller_para, BuildAsset(fundata=readtable("parachiller.csv"),ramp_up=6,ramp_down=-6,startcost=15)); # chiller1
temp_data=readtable("parachiller.csv");
temp_data[:a]=temp_data[:a]+1e-4;
temp_data[:b]=temp_data[:b]+1e-4;
push!(chiller_para, BuildAsset(fundata=temp_data,ramp_up=1.5,ramp_down=-1.5,startcost=20)); # chiller2
push!(chiller_para, BuildAsset(fundata=readtable("parachiller3.csv"),ramp_up=1,ramp_down=-1,startcost=5)); # chiller3

chiller_init=BuildAsset_init[];
push!(chiller_init,BuildAsset_init(status=0));
push!(chiller_init,BuildAsset_init(status=0));
push!(chiller_init,BuildAsset_init(status=1,output=1));
chiller_init[3].status1[24]=1;


abs_para = BuildAsset[];
push!(abs_para, BuildAsset(fundata=readtable("paraabs.csv"),ramp_up=0.25,ramp_down=-0.25,startcost=2));# chiller1

abs_init=BuildAsset_init[];
push!(abs_init,BuildAsset_init(status=0));

KK=3; # number of pieces in piecewise model

a_hru=0.8;

E_storage_para=storage[];
push!(E_storage_para, storage(Emax=2000.0,pmax=500.0,eta_ch=0.93,eta_disch=0.97,soc_min=0.1));

Cool_storage_para=storage[];
push!(Cool_storage_para, storage(Emax=20.0,pmax=5.0,eta_ch=0.94,eta_disch=0.94));

bigM=1e4;
H1=1;
H2=24;
H_t=H2-H1+1;


N_turbine=length(turbine_para);
N_boiler=length(boiler_para);
N_chiller=length(chiller_para);
N_abs=length(abs_para);

N_E_storage=length(E_storage_para);
N_Cool_storage=length(Cool_storage_para);


m = Model(solver=CbcSolver())
@variable(m, turbine_y[1:N_turbine,1:H_t] >=0 ) # gas
@variable(m, turbine_x[1:N_turbine,1:H_t] >=0 ) # electricity
@variable(m, turbine_x_k[1:N_turbine,1:H_t,1:KK] >=0)
@variable(m, turbine_s[1:N_turbine,1:H_t], Bin)
@variable(m, turbine_s_k[1:N_turbine,1:H_t,1:KK], Bin)
@variable(m, turbine_start[1:N_turbine,1:H_t], Bin)

@variable(m, boiler_y[1:N_boiler,1:H_t] >=0) # gas
@variable(m, boiler_x[1:N_boiler,1:H_t] >=0) # heat
@variable(m, boiler_x_k[1:N_boiler,1:H_t,1:KK] >=0)
@variable(m, boiler_s[1:N_boiler,1:H_t], Bin)
@variable(m, boiler_s_k[1:N_boiler,1:H_t,1:KK], Bin)
@variable(m, boiler_start[1:N_boiler,1:H_t], Bin)

@variable(m, chiller_y[1:N_chiller,1:H_t] >=0) # electricity
@variable(m, chiller_x[1:N_chiller,1:H_t] >=0) # cooling heat
@variable(m, chiller_x_k[1:N_chiller,1:H_t,1:KK] >=0)
@variable(m, chiller_s[1:N_chiller,1:H_t], Bin)
@variable(m, chiller_s_k[1:N_chiller,1:H_t,1:KK], Bin)
@variable(m, chiller_start[1:N_chiller,1:H_t], Bin)

@variable(m, abs_y[1:N_abs,1:H_t] >=0) #  wastedheat = f(cooling heat)
@variable(m, abs_x[1:N_abs,1:H_t] >=0) #  cooling heat
@variable(m, abs_x_k[1:N_abs,1:H_t,1:KK] >=0)
@variable(m, abs_s[1:N_abs,1:H_t], Bin)
@variable(m, abs_s_k[1:N_abs,1:H_t,1:KK], Bin)
@variable(m, abs_start[1:N_abs,1:H_t], Bin)

@variable(m, 0 <= E_storage_disch[i=1:N_E_storage,1:H_t] <= E_storage_para[i].pmax)
@variable(m, 0 <= E_storage_ch[i=1:N_E_storage,1:H_t] <= E_storage_para[i].pmax)
@variable(m, E_storage_para[i].Emax * E_storage_para[i].soc_min <= E_storage_state[i=1:N_E_storage,1:H_t] <= E_storage_para[i].Emax * E_storage_para[i].soc_max)

@variable(m, 0 <= Cool_storage_disch[i=1:N_Cool_storage,1:H_t] <= Cool_storage_para[i].pmax)
@variable(m, 0 <= Cool_storage_ch[i=1:N_Cool_storage,1:H_t] <= Cool_storage_para[i].pmax)
@variable(m, Cool_storage_para[i].Emax * Cool_storage_para[i].soc_min <= Cool_storage_state[i=1:N_Cool_storage,1:H_t] <= Cool_storage_para[i].Emax * Cool_storage_para[i].soc_max)

@variable(m, E_gridelecfromgrid[1:H_t]>=0)
@variable(m, E_gridelectogrid[1:H_t]>=0)
@variable(m, Q_HRUheating_in[1:H_t] >=0)
@variable(m, Q_HRUheating_out[1:H_t] >=0)

@variable(m, E_unserve[1:H_t] >=0)
@variable(m, E_dump[1:H_t] >=0)
@variable(m, Heat_unserve[1:H_t] >=0)
@variable(m, Heat_dump[1:H_t] >=0)
@variable(m, Cool_unserve[1:H_t] >=0)
@variable(m, Cool_dump[1:H_t] >=0)


@constraint(m, ElecBalance[t=1:H_t], sum(turbine_x[:,t]) + E_gridelecfromgrid[t] - E_gridelectogrid[t] - sum(chiller_y[:,t]) + sum(E_storage_disch[:,t] - E_storage_ch[:,t])
+ E_unserve[t] - E_dump[t] == parasys[:E_load][t]-parasys[:E_PV][t])
@constraint(m, HeatBalance[t=1:H_t], Q_HRUheating_out[t] + sum(boiler_x[:,t])
+ Heat_unserve[t] - Heat_dump[t] == parasys[:Q_loadheat][t])
@constraint(m, CoolBalance[t=1:H_t], sum(abs_x[:,t]) + sum(chiller_x[:,t]) + sum(Cool_storage_disch[:,t] - Cool_storage_ch[:,t])
+ Cool_unserve[t] - Cool_dump[t] == parasys[:Q_loadcool][t])

E_storage0=zeros(N_E_storage,1);
for i=1:N_E_storage
    E_storage0[i]=0.5*E_storage_para[i].Emax;
end
E_storageend=zeros(N_E_storage,1);
E_storageend[1]=628.87;
@constraint(m, E_storage_init[i=1:N_E_storage], E_storage_state[i,1] == E_storage0[i] + E_storage_para[i].eta_ch * E_storage_ch[i,1]- 1/E_storage_para[i].eta_disch * E_storage_disch[i,1] )
@constraint(m, E_storage_state_constraint[i=1:N_E_storage,t=2:H_t], E_storage_state[i,t] == E_storage_state[i,t-1] + E_storage_para[i].eta_ch * E_storage_ch[i,t]- 1/E_storage_para[i].eta_disch * E_storage_disch[i,t] )
@constraint(m, E_storage_final[i=1:N_E_storage], E_storage_state[i,H_t] >= E_storageend[i])

Cool_storage0=zeros(N_Cool_storage,1);
for i=1:N_Cool_storage
    Cool_storage0[i]=0.5*Cool_storage_para[i].Emax;
end
Cool_storageend=zeros(N_Cool_storage,1);
Cool_storageend[1]=15.647;
@constraint(m, Cool_storage_init[i=1:N_Cool_storage], Cool_storage_state[i,1] == Cool_storage0[i] + Cool_storage_para[i].eta_ch * Cool_storage_ch[i,1]- 1/Cool_storage_para[i].eta_disch * Cool_storage_disch[i,1] )
@constraint(m, Cool_storage_state_constraint[i=1:N_Cool_storage,t=2:H_t], Cool_storage_state[i,t] == Cool_storage_state[i,t-1] + Cool_storage_para[i].eta_ch * Cool_storage_ch[i,t]- 1/Cool_storage_para[i].eta_disch * Cool_storage_disch[i,t] )
@constraint(m, Cool_storage_final[i=1:N_Cool_storage], Cool_storage_state[i,H_t] >= Cool_storageend[i])

@constraint(m, turbineyConsume[i=1:N_turbine,t=1:H_t], turbine_y[i,t] ==turbine_para[i].fundata[:a]' * turbine_x_k[i,t,:] + turbine_para[i].fundata[:b]'*turbine_s_k[i,t,:] )
@constraint(m, turbinexGenerate[i=1:N_turbine,t=1:H_t], turbine_x[i,t] == sum(turbine_x_k[i,t,:]))
@constraint(m, turbinexlower[i=1:N_turbine,t=1:H_t,k=1:KK], turbine_para[i].fundata[:min][k] * turbine_s_k[i,t,k] <= turbine_x_k[i,t,k] )
@constraint(m, turbinexupper[i=1:N_turbine,t=1:H_t,k=1:KK], turbine_para[i].fundata[:max][k] * turbine_s_k[i,t,k] >= turbine_x_k[i,t,k] )
@constraint(m, turbinexstatus[i=1:N_turbine,t=1:H_t], turbine_s[i,t] == sum(turbine_s_k[i,t,:]))
@constraint(m, turbinestartstatus1[i=1:N_turbine,t=1], turbine_start[i,t] >= turbine_s[i,1] - turbine_init[i].status)
@constraint(m, turbinestartstatus[i=1:N_turbine,t=2:H_t], turbine_start[i,t] >= turbine_s[i,t] - turbine_s[i,t-1])
@constraint(m, turbineramp1[i=1:N_turbine,t=1], turbine_init[i].output + turbine_para[i].ramp_down  <= turbine_x[i,t] <= turbine_init[i].output + turbine_para[i].ramp_up )
@constraint(m, turbinerampup[i=1:N_turbine,t=2:H_t], turbine_x[i,t-1] + turbine_para[i].ramp_down  <= turbine_x[i,t] )
@constraint(m, turbinerampdown[i=1:N_turbine,t=2:H_t],  turbine_x[i,t] <= turbine_x[i,t-1] + turbine_para[i].ramp_up )
@constraint(m, turbineslockon1[i=1:N_turbine,t=1],  turbine_para[i].min_on * (turbine_init[i].status1[24] - turbine_s[i,t]) <= sum(turbine_s[i,tau] for tau = 1:t-1) + sum(turbine_init[i].status1[24+t-turbine_para[i].min_on:24]) )
@constraint(m, turbineslockon2[i=1:N_turbine,t=2:turbine_para[i].min_on],  turbine_para[i].min_on * (turbine_s[i,t-1] - turbine_s[i,t]) <= sum(turbine_s[i,tau] for tau = 1:t-1) + sum(turbine_init[i].status1[24+t-turbine_para[i].min_on:24]) )
@constraint(m, turbineslockon[i=1:N_turbine,t=turbine_para[i].min_on+1:H_t], turbine_para[i].min_on * (turbine_s[i,t-1] - turbine_s[i,t]) <= sum(turbine_s[i,tau] for tau = t - turbine_para[i].min_on:t-1))



@constraint(m, boileryConsume[i=1:N_boiler,t=1:H_t], boiler_y[i,t] ==boiler_para[i].fundata[:a]' * boiler_x_k[i,t,:] + boiler_para[i].fundata[:b]'*boiler_s_k[i,t,:] )
@constraint(m, boilerxGenerate[i=1:N_boiler,t=1:H_t], boiler_x[i,t] == sum(boiler_x_k[i,t,:]))
@constraint(m, boilerxlower[i=1:N_boiler,t=1:H_t,k=1:KK], boiler_para[i].fundata[:min][k] * boiler_s_k[i,t,k] <= boiler_x_k[i,t,k] )
@constraint(m, boilerxupper[i=1:N_boiler,t=1:H_t,k=1:KK], boiler_para[i].fundata[:max][k] * boiler_s_k[i,t,k] >= boiler_x_k[i,t,k] )
@constraint(m, boilerxstatus[i=1:N_boiler,t=1:H_t], boiler_s[i,t] == sum(boiler_s_k[i,t,:]))
@constraint(m, boilerstartstatus1[i=1:N_boiler,t=1], boiler_start[i,t] >= boiler_s[i,1] - boiler_init[i].status)
@constraint(m, boilerstartstatus[i=1:N_boiler,t=2:H_t], boiler_start[i,t] >= boiler_s[i,t] - boiler_s[i,t-1])
@constraint(m, boilerramp1[i=1:N_boiler,t=1], boiler_init[i].output + boiler_para[i].ramp_down  <= boiler_x[i,t] <= boiler_init[i].output + boiler_para[i].ramp_up )
@constraint(m, boilerrampup[i=1:N_boiler,t=2:H_t], boiler_x[i,t-1] + boiler_para[i].ramp_down  <= boiler_x[i,t] )
@constraint(m, boilerrampdown[i=1:N_boiler,t=2:H_t],  boiler_x[i,t] <= boiler_x[i,t-1] + boiler_para[i].ramp_up )



@constraint(m, chilleryConsume[i=1:N_chiller,t=1:H_t], chiller_y[i,t] ==chiller_para[i].fundata[:a]' * chiller_x_k[i,t,:] + chiller_para[i].fundata[:b]'*chiller_s_k[i,t,:] )
@constraint(m, chillerxGenerate[i=1:N_chiller,t=1:H_t], chiller_x[i,t] == sum(chiller_x_k[i,t,:]))
@constraint(m, chillerxlower[i=1:N_chiller,t=1:H_t,k=1:KK], chiller_para[i].fundata[:min][k] * chiller_s_k[i,t,k] <= chiller_x_k[i,t,k] )
@constraint(m, chillerxupper[i=1:N_chiller,t=1:H_t,k=1:KK], chiller_para[i].fundata[:max][k] * chiller_s_k[i,t,k] >= chiller_x_k[i,t,k] )
@constraint(m, chillerxstatus[i=1:N_chiller,t=1:H_t], chiller_s[i,t] == sum(chiller_s_k[i,t,:]))
@constraint(m, chillerstartstatus1[i=1:N_chiller,t=1], chiller_start[i,t] >= chiller_s[i,1] - chiller_init[i].status)
@constraint(m, chillerstartstatus[i=1:N_chiller,t=2:H_t], chiller_start[i,t] >= chiller_s[i,t] - chiller_s[i,t-1])
@constraint(m, chillerramp1[i=1:N_chiller,t=1], chiller_init[i].output + chiller_para[i].ramp_down  <= chiller_x[i,t] <= chiller_init[i].output + chiller_para[i].ramp_up )
@constraint(m, chillerrampup[i=1:N_chiller,t=2:H_t], chiller_x[i,t-1] + chiller_para[i].ramp_down  <= chiller_x[i,t] )
@constraint(m, chillerrampdown[i=1:N_chiller,t=2:H_t],  chiller_x[i,t] <= chiller_x[i,t-1] + chiller_para[i].ramp_up )



@constraint(m, absyConsume[i=1:N_abs,t=1:H_t], abs_y[i,t] ==abs_para[i].fundata[:a]' * abs_x_k[i,t,:] + abs_para[i].fundata[:b]'*abs_s_k[i,t,:] )
@constraint(m, absxGenerate[i=1:N_abs,t=1:H_t], abs_x[i,t] == sum(abs_x_k[i,t,:]))
@constraint(m, absxlower[i=1:N_abs,t=1:H_t,k=1:KK], abs_para[i].fundata[:min][k] * abs_s_k[i,t,k] <= abs_x_k[i,t,k] )
@constraint(m, absxupper[i=1:N_abs,t=1:H_t,k=1:KK], abs_para[i].fundata[:max][k] * abs_s_k[i,t,k] >= abs_x_k[i,t,k] )
@constraint(m, absxstatus[i=1:N_abs,t=1:H_t], abs_s[i,t] == sum(abs_s_k[i,t,:]))
@constraint(m, absstartstatus1[i=1:N_abs,t=1], abs_start[i,t] >= abs_s[i,1] - abs_init[i].status)
@constraint(m, absstartstatus[i=1:N_abs,t=2:H_t], abs_start[i,t] >= abs_s[i,t] - abs_s[i,t-1])
@constraint(m, absramp1[i=1:N_abs,t=1], abs_init[i].output + abs_para[i].ramp_down  <= abs_x[i,t] <= abs_init[i].output + abs_para[i].ramp_up )
@constraint(m, absrampup[i=1:N_abs,t=2:H_t], abs_x[i,t-1] + abs_para[i].ramp_down  <= abs_x[i,t] )
@constraint(m, absrampdown[i=1:N_abs,t=2:H_t],  abs_x[i,t] <= abs_x[i,t-1] + abs_para[i].ramp_up )



@constraint(m, wastedheat[t=1:H_t], Q_HRUheating_in[t] + abs_y[t] == sum(turbine_y[1:2,t]-turbine_x[1:2,t]/293.1) )
@constraint(m, HRUlimit[t=1:H_t], Q_HRUheating_out[t] <= a_hru * Q_HRUheating_in[t] )


#s @constraint(m, power_onoff_constraint[n=1:N,t=1:H_t], power[n,t] <= 5*status[n,t])
@objective(m, Min,
           E_gridelecfromgrid' * parasys[:lambda_elec_fromgrid][1:H_t] + E_gridelectogrid' * parasys[:lambda_elec_togrid][1:H_t]
+ (ones(1,2)*turbine_y[1:2,:]*parasys[:lambda_gas][1:H_t])[1]
+ turbine_y[3,:]'*parasys[:lambda_diesel][1:H_t]
+ (ones(1,N_boiler)*boiler_y*parasys[:lambda_gas][1:H_t])[1]
 + sum(sum(turbine_start[i,:]) * turbine_para[i].startcost for i=1:N_turbine)
+ sum(sum(boiler_start[i,:]) * boiler_para[i].startcost for i=1:N_boiler)
+ sum(sum(chiller_start[i,:]) * chiller_para[i].startcost for i=1:N_chiller)
+ sum(sum(abs_start[i,:]) * abs_para[i].startcost for i=1:N_abs)
+ sum(E_unserve+E_dump+Heat_unserve+Heat_dump+Cool_unserve+Cool_dump)*bigM
)

writeLP(m, "Building_opt_julia3.lp",genericnames=false)

tic()
sol_status=solve(m)
toc()


ob=getobjectivevalue(m)

sol_E_dump=getvalue(E_dump)
sol_E_unserve=getvalue(E_unserve)
sol_Cool_dump=getvalue(Cool_dump)
sol_Cool_unserve=getvalue(Cool_unserve)
sol_Heat_dump=getvalue(Heat_dump)
sol_Heat_unserve=getvalue(Heat_unserve)

println("Objective value: ", ob)

println("E_dump = ", maximum(sol_E_dump))
println("E_unserve = ", maximum(sol_E_unserve))
println("Heat_dump = ", maximum(sol_Heat_dump))
println("Heat_unserve = ", maximum(sol_Heat_unserve))
println("Cool_dump = ", maximum(sol_Cool_dump))
println("Cool_unserve = ", maximum(sol_Cool_unserve))

# print(m)
solution=DataFrame(Electric_Utility = getvalue(E_gridelecfromgrid) -getvalue(E_gridelectogrid), Solar=parasys[:E_PV],
Fuel_cell=getvalue(turbine_x[1,:]), turbine=getvalue(turbine_x[2,:]), Diesel=getvalue(turbine_x[3,:]),
HeatRecover=getvalue(Q_HRUheating_out),Battery=getvalue(E_storage_disch[1,:]) - getvalue(E_storage_ch[1,:]),
Boiler1=getvalue(boiler_x[1,:]), boiler2=getvalue(boiler_x[2,:]),
Chiller1=getvalue(chiller_x[1,:]), Chiller2=getvalue(chiller_x[2,:]),Chiller3=getvalue(chiller_x[3,:]),
Abschiler=getvalue(abs_x[1,:]), Coolstorage=getvalue(Cool_storage_disch[1,:]) - getvalue(Cool_storage_ch[1,:]))
writetable("output_write.csv",solution)

solutiony=DataFrame(Fuel_cell=getvalue(turbine_y[1,:]), turbine=getvalue(turbine_y[2,:]), Diesel=getvalue(turbine_y[3,:]),
HeatRecover=getvalue(Q_HRUheating_in),Boiler1=getvalue(boiler_y[1,:]), boiler2=getvalue(boiler_y[2,:]),
Chiller1=getvalue(chiller_y[1,:]), Chiller2=getvalue(chiller_y[2,:]),Chiller3=getvalue(chiller_y[3,:]),
Abschiler=getvalue(abs_y[1,:]),Battstate=getvalue(E_storage_state[1,:]),Coolstate=getvalue(Cool_storage_state[1,:]))
writetable("output_write_y.csv",solutiony)
