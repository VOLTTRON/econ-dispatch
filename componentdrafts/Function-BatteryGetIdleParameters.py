
# coding: utf-8

# In[ ]:

def GetIdleParameters(TrainingData, Capacity):
    from scipy import stats
    import numpy as np
    
    Time= TrainingData['Time'].values
    SOC=TrainingData['SOC'].values
    Rows=len(Time)
    PrevTime = []
    CurrTime = []
    CurrSOC=[]
    PrevSOC=[]
    Y=[]
    X=[]
    for i in range(0,Rows+1):
        if i>1:
            PrevTime.append(Time[i-2])
            CurrTime.append(Time[i-1])
            CurrSOC.append(SOC[i-1])
            PrevSOC.append(SOC[i-2])
    for i in range(0,Rows-1):
        Y.append((CurrSOC[i]-PrevSOC[i])/((CurrTime[i]-PrevTime[i])*24))
        X.append(CurrSOC[i])
        
    slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
    C = [slope, intercept]
    return C

