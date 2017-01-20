
# coding: utf-8

# In[ ]:

def GetRegressionHeatRecovery(TrainingData):
    from scipy import stats
    import numpy as np
    
    Twi= TrainingData['T_wi'].values
    Two= TrainingData['T_wo'].values
    Texi= TrainingData['T_ex_i'].values
    Speed=TrainingData['Speed'].values
    MFRw=TrainingData['MFR_water'].values
    Proportional= Speed/MFRw*(Texi-Twi)
    Output=Two-Twi
    
    x = Proportional
    y = Output
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    C = [intercept, slope]
    return C                 
                      

