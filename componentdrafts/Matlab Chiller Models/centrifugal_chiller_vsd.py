import json
import numpy as np


def train():
    # *****************     Training Module       *****************
    # ***** Centrifugal Chiller (with variable speed drive control) ******

    # This module reads the historical data on temperatures (in Fahrenheit), inlet power to the
    # chiller (in kW) and outlet cooling load (in cooling ton) then, converts
    # the data to proper units which then will be used for model training. At
    # the end, regression coefficients will be written to an excel file

    # ******** Reading data from dataset ************
    with open('CH-Cent-VSD-Historical-Data.json', 'r') as f:
        historical_data = json.load(f)

    Tcho = historical_data["Tcho(F)"]# chilled water supply temperature in F
    Tcdi = historical_data["Tcdi(F)"]# condenser water temperature (outlet from heat rejection and inlet to chiller) in F
    Qch = historical_data["Qch(tons)"]# chiller cooling output in Tons of cooling
    P = historical_data["P(kW)"]# chiller power input in kW
    
    i = len(Tcho)
    U = np.ones(i)

    NameplateAvailable = True #User input of maximum chiller capacity, if available
    if NameplateAvailable:
        Qchmax_Tons= 500  #Chiller capacity in cooling tons
        Qchmax = Qchmax_Tons*12000/3412
    else:
        Qchmax = max(Qch)

    # *********************************

    COP = np.zeros(i) # Chiller COP
    x1 = np.zeros(i)
    x2 = np.zeros(i)
    x3 = np.zeros(i)
    x4 = np.zeros(i)
    y = np.zeros(i)
    #tstd = zeros(24,1) # time stamp

    for a in range(i):
        Tcho[a] = (Tcho[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
        Tcdi[a] = (Tcdi[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
        Qch[a] = Qch[a] * 12000 / 3412 # Converting tons to kW
        COP[a] = Qch[a] / P[a]

    for a in range(i):
        x1[a] = Tcho[a] / Qch[a]
        x2[a] = Tcho[a] / Qchmax
        x3[a] = (Tcdi[a] - Tcho[a]) / (Tcdi[a] * Qch[a])
        x4[a] = (((1 / COP[a]) + 1) * Qch[a]) / Tcdi[a]
        y[a] = ((((1 / COP[a]) + 1) * Tcho[a]) / Tcdi[a]) - 1

    #*******Multiple Linear Regression***********
    XX = np.column_stack((U,x1,x2,x3,x4))#matrix of predictors
    AA, resid, rank, s = np.linalg.lstsq(XX, y)
    #********************************************

    return Qchmax, AA


def predict():
    #*****************     Deployment Module       *****************
    # ***** Centrifugal Chiller (with variable speed drive control) ******

    # Regression models were built separately (Training Module) and
    # therefore regression coefficients are available. Also, forecasted values
    # for Chiller cooling output were estimated from building load predictions.
    # This code is meant to be used for 24 hours ahead predictions.
    # The code creates an excel file and writes
    # the results on it along with time stamps

    # State Variable Inputs
    Tcho = 44 #Chilled water temperature setpoint outlet from chiller
    Tcdi = 75 # Condenser water temperature inlet temperature to chiller from condenser in F
    # Note that this fixed value of 75F is a placeholder.  We will ultimately
    # need a means of forecasting the condenser water inlet temperature.
    Qch_kW = 1758.5 #building cooling load ASSIGNED TO THIS CHILLER in kW

    # This is never used why is it here?
    Qch = Qch_kW * 3412 / 12000 # Building Cooling Load in Tons


    # ********* Gordon-Ng model coefficients ********
    Qchmax, (a0, a1, a2, a3, a4) = train()
    # ***********************************************

    Tcho_K = (Tcho - 32) / 1.8 + 273.15#Converting F to Kelvin
    Tcdi_K = (Tcdi - 32) / 1.8 + 273.15#Converting F to Kelvin


    COP = ((Tcho_K / Tcdi_K) - a4 * (Qch_kW / Tcdi_K)) / ((a0 + (a1 + a2 * (Qch_kW / Qchmax)) * (Tcho_K / Qch_kW) + a3 * ((Tcdi_K - Tcho_K) / (Tcdi_K * Qch_kW)) + 1) - ((Tcho_K / Tcdi_K) - a4 * (Qch_kW / Tcdi_K)))
    #Coefficient of Performance(COP) of chiller from regression

    P_Ch_In = Qch_kW / COP #Chiller Electric Power Input in kW


if __name__ == '__main__':
    predict()
