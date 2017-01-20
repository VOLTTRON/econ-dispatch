
# coding: utf-8

# In[ ]:

get_ipython().magic(u'matplotlib inline')

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import * 
from IPython.display import Javascript, display
from collections import OrderedDict



InverterCurve = pd.read_csv('C:\Users\d3x836\Desktop\PNNL-Nick Fernandez\GMLC\InverterCurve.csv', header=0)
InverterCurveArray=pd.DataFrame(InverterCurve)
PLFvals= InverterCurve['PLF'].values
EffVals=InverterCurve['Efficiency'].values
ElecIn=10
InverterCapacity=150.0
ElecOut=InverterElecOut(PLFvals,EffVals,ElecIn,InverterCapacity)
print ElecOut

