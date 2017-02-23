import json
import numpy as np


def train():
    #*************     Training Module       ********
    # *************   Absorption Chiller    *********
    # This module reads the historical data on temperatures (in Fahrenheit), inlet heat to the
    # chiller (in mmBTU/hr) and outlet cooling load (in cooling ton) then, converts
    # the data to proper units which then will be used for model training. At
    # the end, regression coefficients will be written to a file

    # ******** Reading data from dataset ************
    with open('CH-Abs-Historical-Data.json', 'r') as f:
        historical_data = json.load(f)

    
    Tcho = historical_data["Tcho(F)"]# chilled water supply temperature in F
    Tcdi = historical_data["Tcdi(F)"]# inlet temperature from condenser in F
    Tgeni = historical_data["Tgei(F)"]# generator inlet water temperature in F
    Qch = historical_data["Qch(tons)"]# chiller cooling output in cooling Tons
    Qin = historical_data["Qin(MMBtu/h)"]# chiller heat input in mmBTU/hr
    i = len(Tcho)
    U = np.ones(i)

    # *********************************

    COP = np.zeros(i) # Chiller COP
    x1 = np.zeros(i)
    y = np.zeros(i)


    for a in range(i):
        Tcho[a] = (Tcho[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
        Tcdi[a] = (Tcdi[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
        Tgeni[a] = (Tgeni[a] - 32) / 1.8 + 273.15#Converting F to Kelvin
        Qch[a] = 3.517 * Qch[a]#Converting cooling tons to kW
        Qin[a] = 293.1 * Qin[a]#Converting mmBTU/hr to kW
        COP[a] = float(Qch[a]) / float(Qin[a])

    for a in range(i):
        x1[a] = float(Tcdi[a]) / float(Tgeni[a])
        y[a] = ((Tgeni[a] - Tcdi[a]) / float((Tgeni[a] * COP[a])) - ((Tgeni[a] - Tcho[a]) / float(Tcho[a]))) * Qch[a]

    #*******Multiple Linear Regression***********
    XX = np.column_stack((U, x1))
    AA, resid, rank, s = np.linalg.lstsq(XX, y)
    #********************************************

    return AA


def predict():
    #*************     Deployment Module       ********
    # *************    Absorption Chiller    *********

    # Regression models were built separately (Training Module) and
    # therefore regression coefficients are available. Heat input to the chiller generator
    # is assumed to be known and this model predicts the chiller cooling output.
    # This code is meant to be used for 4 hours ahead predictions.
    # The code creates an excel file and writes
    # the results on it along with time stamps.

    # Dynamic Inputs
    Tcho = 45.8 #Chilled water temperature setpoint outlet from absorption chiller
    Tcdi = 83.7 # Condenser water temperature inlet temperature to absorption chiller from heat rejection in F
    Tgeni = 335 # Generator inlet temperature (hot water temperature inlet to abs chiller) in F
    Qin = 8.68# heat input to the generator in mmBTU/hr


    # ********* Gordon-Ng model coefficients ********
    a0, a1 = train()
    # ***********************************************

    Tcho = (Tcho - 32) / 1.8 + 273.15 #Converting F to Kelvin
    Tcdi = (Tcdi - 32) / 1.8 + 273.15 #Converting F to Kelvin
    Tgeni = (Tgeni - 32)/ 1.8 + 273.15 #Converting F to Kelvin
    Qin = 293.1 * Qin #Converting mmBTU/hr to kW

    Qch = (Qin * ((Tgeni - Tcdi) / Tgeni) - a0 - a1 * (Tcdi / Tgeni)) / ((Tgeni - Tcho) / Tcho)
    Qch = Qch / 3.517 #Converting kW to cooling ton


if __name__ == '__main__':
    predict()
