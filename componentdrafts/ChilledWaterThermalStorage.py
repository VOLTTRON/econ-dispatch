# coding: utf-8

# In[ ]:

import sqlite3
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.matlib

# Tank Volume in L
TankVolume = 37800.0

# Tank Height in m
TankHeight = 2.0

InsulatedTank = False

Fluid = 'water'

timestep = 60

# Flow rate of chilled water to the tank from the chiller, in kg/s
MFR_chW = 50.4

# Temperature of chilled water from the chiller, in degrees C
T_chW = 7

# Flow rate of chilled water to the tank from the absorption chiller, in kg/s
MFR_abW = 10

# Temperature of chilled water from the absorption chiller, in degrees C
T_abW = 10

# Flow rate of chilled water to the building loads from the tank
MFR_chwBldg = 30

# return chilled water temperature from bulding in degress C
T_chwBldgReturn = 14

# initialization of temperatures in tank
Nodes = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6]


for i in range(1,2):
    Nodes = getNodeTemperatures(timestep, MFR_chW, T_chW, MFR_abW, T_abW, MFR_chwBldg, T_chwBldgReturn, Nodes, TankVolume, TankHeight, Fluid, InsulatedTank)
    print Nodes
