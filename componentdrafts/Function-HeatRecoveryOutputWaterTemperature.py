# coding: utf-8

import math

def HeatRecoveryWaterOut(PrimeMoverFluid, HotWaterFluid, Method, T_water_i, T_ex_i, MFR_water, MFR_ex, CurrSpeed, TrainingData, HeatExchagerArea, DefaultEffectiveness):

    if PrimeMoverFluid == 'FlueGas':
        cp_ex=1002 #j/kg-K
    elif PrimeMoverFluid == 'Water':
        cp_ex=4186 #j/kg-K
    elif PrimeMoverFluid == 'GlycolWater30':
        cp_ex=3913 #j/kg-K
    elif PrimeMoverFluid == 'GlycolWater50':
        cp_ex=3558 #j/kg-K
    else:
        cp_ex=4186 #j/kg-K

    if HotWaterLoopFluid == 'Water':
        cp_water=4186 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater30':
        cp_water=3913 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater50':
        cp_water=3558 #j/kg-K
    else:
        cp_water=4186 #j/kg-K

    C_ex = MFR_ex * cp_ex
    C_water = MFR_water * cp_water
    C_min = min(C_ex, C_water)
    C_max = max(C_ex, C_water)
    C_r = C_min / C_max

    if Method == 1:
        U = 650 #w/m2-K
        UA = U * HeatExchagerArea #W-K
        NTU = UA / C_min
        Eff = (1 - math.exp(-1 * NTU * (1 - C_r))) / (1 - C_r * math.exp(-1 * NTU * (1 - C_r))) #relation for concentric tube in counterflow
        T_water_o = T_water_i + Eff * C_min / C_water * (T_ex_i - T_water_i)
    elif Method == 2:
        Eff = DefaultEffectiveness
        T_water_o = T_water_i + Eff * C_min / C_water * (T_ex_i - T_water_i)
    elif Method == 3:
        T_water_o = C[0] + T_water_i + C[1] * CurrSpeed / MFR_water * (T_ex_i - T_water_i)

    return T_water_o
