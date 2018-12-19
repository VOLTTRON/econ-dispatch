import datetime
import json

import pandas as pd
import numpy as np

data = pd.read_csv('Burrstone_2017_YTD_data.csv', 
                   header=[0,1], 
                   parse_dates={('timestamp', 'datetime'):[0,1]}, 
                   na_values=[-99])  # na_values=[-99,0,-0]
data.columns = pd.MultiIndex(levels=[[l.strip() for l in data.columns.levels[0]], 
                                     [l.strip() for l in data.columns.levels[1]]], 
                             labels=data.columns.labels)
units_lookup = {var: unit for var, unit in data.columns}
data.columns = data.columns.levels[0][data.columns.labels[0]]

decision_variables = ['timestamp', 
                      'TAO', 
                      'WCG1_cum', 
                      'WCOL_ex', 
                      'WCOL_im', 
                      'WCG2_cum', 
                      'WCG3_cum', 
                      'WHSP1_ex', 
                      'WHSP2_ex', 
                      'WHSP1_im', 
                      'WHSP2_im', 
                      'WCG4_cum', 
                      'WHOM_ex', 
                      'WHOM_im', 
                      'FST', 
                      'FW', 
                      'TL', 
                      'TE']

inds = pd.isnull(data[decision_variables]).any(1).nonzero()[0]
data.loc[inds,decision_variables].to_csv('data_nulls.csv', na_rep='nan', index=False)

inds = (data[decision_variables[2:]] == 0).any(1).nonzero()[0]
data[decision_variables].loc[inds, :].to_csv('data_zeros.csv', na_rep='nan', index=False)

data2 = pd.DataFrame({
    'temperature': data['TAO'],
    'college_elec_load': data['WCG1_cum'] + data['WCOL_ex'] - data['WCOL_im'],
    'hospital_elec_load': data['WCG2_cum'] + data['WCG3_cum'] + data['WHSP1_ex'] + data['WHSP2_ex'] - data['WHSP1_im'] - data['WHSP2_im'],
    'home_elec_load': data['WCG4_cum'] + data['WHOM_ex'] - data['WHOM_im'],
    'steam_load': data['FST'],  # lb/hr  # *0.001194, for mmBTU/hr  # *0.3497 for kWh
    'heat_load': data['FW']*(data['TL'] - data['TE'])*0.0005  # mmBTU/hr  # *0.1465 for kWh
})
data2.index = data['timestamp']

nan_filter = ~pd.isnull(data2).any(1)
assert np.all(nan_filter == ~pd.isnull(data2['temperature']))
data3 = data2.loc[nan_filter,:].copy()

for var in ['college_elec_load', 'hospital_elec_load', 'home_elec_load', 'steam_load', 'heat_load']:
    data3.loc[(data3.loc[:,var] == 0), var] = np.nan
data3.to_csv('burrstone_loads.csv', na_rep='')

units_lookup['college_elec_load'] = units_lookup['WCG1_cum']  # kWh
assert units_lookup['college_elec_load'] == units_lookup['WCOL_ex']
assert units_lookup['college_elec_load'] == units_lookup['WCOL_im']
units_lookup['hospital_elec_load'] = units_lookup['WCG2_cum']# 'kWh'
assert units_lookup['hospital_elec_load'] == units_lookup['WCG3_cum']
assert units_lookup['hospital_elec_load'] == units_lookup['WHSP1_ex']
assert units_lookup['hospital_elec_load'] == units_lookup['WHSP2_ex']
assert units_lookup['hospital_elec_load'] == units_lookup['WHSP1_im']
assert units_lookup['hospital_elec_load'] == units_lookup['WHSP2_im']
units_lookup['home_elec_load'] = units_lookup['WCG4_cum']  # kWh
assert units_lookup['home_elec_load'] == units_lookup['WHOM_ex']
assert units_lookup['home_elec_load'] == units_lookup['WHOM_im']
units_lookup['steam_load'] = units_lookup['FST']  # 'lb/hr'
units_lookup['heat_load'] = 'mmBTU/hr'  # 'mmBTU/hr'

with open('units.json', 'w') as f:
    json.dump(units_lookup, f)
