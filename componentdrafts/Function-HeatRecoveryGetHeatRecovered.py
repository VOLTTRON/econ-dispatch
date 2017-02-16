# coding: utf-8

def GetHeatRecovered(HotWaterLoopFluid,T_water_i,TWaterOut,MFR_water):
    if HotWaterLoopFluid == 'Water':
        cp_water = 4186 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater30':
        cp_water = 3913 #j/kg-K
    elif HotWaterLoopFluid == 'GlycolWater50':
        cp_water = 3558 #j/kg-K
    else:
        cp_water = 4186 #j/kg-K

    HeatRecovered = MFR_water * cp_water * (TWaterOut - T_water_i) / 1000 #kW
    return HeatRecovered
