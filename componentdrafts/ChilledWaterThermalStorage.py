
# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib

TankVolume = 37800.0 # Tank Volume in L
TankHeight = 2.0 # Tank Height in m
InsulatedTank ='No'
Fluid = 'water'
timestep = 60
MFR_chW = 50.4 # Flow rate of chilled water to the tank from the chiller, in kg/s
T_chW = 7 # Temperature of chilled water from the chiller, in degrees C
MFR_abW = 10 # Flow rate of chilled water to the tank from the absorption chiller, in kg/s
T_abW = 10 # Temperature of chilled water from the absorption chiller, in degrees C
MFR_chwBldg = 30 # Flow rate of chilled water to the building loads from the tank
T_chwBldgReturn = 14 # return chilled water temperature from bulding in degress C

Nodes = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6] # initialization of temperatures in tank


for i in range(1,2):
    Nodes = getNodeTemperatures (timestep, MFR_chW, T_chW, MFR_abW, T_abW, MFR_chwBldg,T_chwBldgReturn,Nodes,TankVolume,TankHeight,Fluid,InsulatedTank)
    print Nodes

