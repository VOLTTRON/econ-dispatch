
# coding: utf-8

# In[ ]:

def getTrainingMFR(TrainingData):
    #This function is used by the Desiccant Wheel program to estimate the delta T across the regeneration coil for the purpose of calculating the hot water mass flow rate in the regeneration coil
    # This funciton is used only when historical data for the regneration coil outlet tempreature is available.
    
    import numpy as np
    from scipy import stats
    import CoolProp
    from CoolProp.HumidAirProp import HAPropsSI
    import statsmodels.api as sm
    
    T_OA= TrainingData['T_OA'].values # Outdoor air temperature sensor; convert to degrees C
    RH_OA= TrainingData['RH_OA'].values # Outdoor air relative humudity (fraction)
    Vlv=TrainingData['Regen_Valve'].values # regeneration coil valve command (fraction)
    T_hw=TrainingData['T_HW'].values # Hot water inlet temperature to regeneration coil [C]
    T_hw_out=TrainingData['T_HW_out'].values  # Hot water outlet temperature from regeneration coil [C]
    
    h_OA_fan=[]
    delta_T=[] # hot water temperature drop across regneration coil [C]
    
    for i in range(0,Rows):
        if Fan[i]>0 and Vlv[i]>0:
            h_OAx= HAPropsSI('H','T',(T_OA[i]+273.15),'P',101325,'R',RH_OA[i]/100); #calculate outdoor air enthalpy 
            h_OA_fan.append(h_OAx)
            delta_T.append(T_hw[i]-T_hw_out[i])
    
    RegressionArray= [h_OA_fan] #single variabel regression based on outdoor air enthalpy
    y = delta_T
    x = RegressionArray

    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
       
    Coefficients=results.params
    return Coefficients
    

