import json
import numpy as np


def train():
    # This module reads the historical data on boiler heat output and
    # gas heat input both in mmBTU/hr then, converts
    # the data to proper units which then will be used for model training. At
    # the end, regression coefficients will be written to a file

    # ******** Reading data from dataset ************
    with open("Boiler-Historical-Data.json", 'r') as f:
        historical_data = json.load(f)

    # boiler gas input in mmBTU
    # Note from Nick Fernandez: Most sites will not have metering for gas inlet
    # to the boiler.  I'm creating a second option to use a defualt boiler
    # curve
    Gbp = historical_data["boiler_gas_input"]

    # boiler heat output in mmBTU
    Qbp = historical_data["boiler_heat_output"]

    i = len(Gbp)
    U = np.ones(i)

    # ****** Static Inputs (Rating Condition + Natural Gas Heat Content *******
    Qbprated = 60 #boiler heat output at rated condition - user input (mmBtu)
    Gbprated = 90 #boiler gas heat input at rated condition - user input (mmBtu)
    #**************************************************************************

    xbp = np.zeros(i)
    xbp2 = np.zeros(i)
    xbp3 = np.zeros(i)
    xbp4 = np.zeros(i)
    xbp5 = np.zeros(i)
    ybp = np.zeros(i)

    for a in range(i):
        xbp[a] = Qbp[a] / Qbprated
        xbp2[a] = xbp[a]**2
        xbp3[a] = xbp[a]**3
        xbp4[a] = xbp[a]**4
        xbp5[a] = xbp[a]**5
        ybp[a] = (Qbp[a] / Gbp[a]) / (float(Qbprated) / float(Gbprated))

    #*******Multiple Linear Regression***********
    XX = np.column_stack((U, xbp, xbp2, xbp3, xbp4, xbp5))#matrix of predictors
    AA, resid, rank, s = np.linalg.lstsq(XX, ybp)
    #********************************************
    
    return AA


def predict():
    # Building heating load assigned to Boiler
    Qbp = 55
    
    # Boiler Nameplate parameters (User Inputs)
    Qbprated = 60 #mmBtu/hr
    Gbprated = 90 # mmBtu/hr
    GasInputSubmetering = True #Is metering of gas input to the boilers available?  If not, we can't build a regression, and instead will rely on default boiler part load efficiency curves
    #************************************************************
    HC = 0.03355 #NG heat Content 950 Btu/ft3 is assumed
    # **************************************************************************
    
    
    if GasInputSubmetering:
        # ********* 5-degree polynomial model coefficients from training*****
        a0, a1, a2, a3, a4, a5 = train()
    else:
        # Use part load curve for 'atmospheric' boiler from http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.553.4931&rep=rep1&type=pdf
        a0 = 0.6978
        a1 = 3.3745
        a2 = -15.632
        a3 = 32.772
        a4 = -31.45
        a5 = 11.268
    
    
    if Qbp > Qbprated:
        Qbp = Qbprated
    
    xbp = Qbp / Qbprated # part load ratio
    ybp = a0 + a1*xbp + a2*(xbp)**2 + a3*(xbp)**3 + a4*(xbp)**4 + a5*(xbp)**5# relative efficiency (multiplier to ratred efficiency)
    Gbp = (Qbp * Gbprated) / (ybp * Qbprated)# boiler gas heat input in mmBtu
    FC = Gbp / HC #fuel consumption in cubic meters per hour
    

if __name__ == '__main__':
    predict()
