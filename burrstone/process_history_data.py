import datetime
import pandas as pd
import numpy as np

decision_variables = ['timestamp',
                      'TAO',
                      'WCOL_ex',
                      'WCOL_im',
                      'WHSP1_ex',
                      'WHSP2_ex',
                      'WHSP1_im',
                      'WHSP2_im',
                      'WHOM_ex',
                      'WHOM_im',
                      #   'WCG1_cum',
                      #   'WCG2_cum',
                      #   'WCG3_cum',
                      #   'WCG4_cum',
                      'WCOL_gen',
                      'WHSP_gen',
                      'WHOM_gen',
                      'FST',
                      'FW',
                      'TL',
                      'TE']

accumulation_variables = ['WCOL_ex',
                          'WCOL_im',
                          'WHSP1_ex',
                          'WHSP2_ex',
                          'WHSP1_im',
                          'WHSP2_im',
                          'WHOM_ex',
                          'WHOM_im']

facilities = {
    'college': {
        # 'generators': ['WCG1_cum'],
        'generators': ['WCOL_gen'],
        'utility': ['WCOL_{}']
    },
    'hospital': {
        # 'generators': ['WCG2_cum', 'WCG3_cum'],
        'generators': ['WHSP_gen'],
        'utility': ['WHSP1_{}', 'WHSP2_{}']
    },
    'home': {
        # 'generators': ['WCG4_cum'],
        'generators': ['WHOM_gen'],
        'utility': ['WHOM_{}']
    }
}


def derive_variables(row={}):
    """ Recombine measurements into useful data"""
    result = {}
    result['timestamp'] = row['timestamp']
    if pd.isnull(row['TAO']):
        result['temperature'] = np.nan
    else:
        result['temperature'] = row['TAO']
    for facility, definition in facilities.items():
        vars = row[definition['generators']
                   + [u.format('ex') for u in definition['utility']]
                   + [u.format('im') for u in definition['utility']]]
        if np.any([pd.isnull(v) for v in vars]):
            result[facility+'_elec_load'] = np.nan
        else:
            generated = sum(row[definition['generators']])
            exported = sum(row[[u.format('im')
                           for u in definition['utility']]])
            imported = sum(row[[u.format('ex')
                           for u in definition['utility']]])
            result[facility+'_elec_load'] = generated+imported-exported
    if pd.isnull(row['FST']):
        result['steam_load'] = np.nan
    else:
        # lb/hr; *0.001194 for mmBTU/hr; *0.3497 for kWh
        result['steam_load'] = row['FST']
    if np.any([pd.isnull(v) for v in row[['FW', 'TL', 'TE']]]):
        result['heat_load'] = np.nan
    else:
        # mmBTU/hr; *0.1465 for kWh
        result['heat_load'] = row['FW']*(row['TL'] - row['TE'])*0.0005
        if result['heat_load'] < 0:
            result['heat_load'] = np.nan

    return pd.Series(result)


# Load raw data
data = pd.read_csv('Burrstone_2017_YTD_data.csv',
                   header=[0, 1],
                   parse_dates={('timestamp', 'datetime'): [0, 1]},
                   na_values=[-99])  # na_values=[-99, 0, -0]
data.columns = \
    pd.MultiIndex(levels=[[l.strip() for l in data.columns.levels[0]],
                          [l.strip() for l in data.columns.levels[1]]],
                  labels=data.columns.labels)
units_lookup = {var: unit for var, unit in data.columns}
data.columns = data.columns.levels[0][data.columns.labels[0]]

# Transform energy accumulation variables into estimates of power
td = (datetime.timedelta(hours=1)/data['timestamp'].diff())[1:]
data2 = data[decision_variables].copy()
# data2[decision_variables[2:-4]] = \
#     data2[decision_variables[2:-4]].diff().multiply(td, axis=0)
data2[decision_variables[2:-7]] = \
    data2[decision_variables[2:-7]].diff().multiply(td, axis=0)

# Apply transform
data3 = data2.apply(derive_variables, axis=1)

# Output
data3.to_csv('burrstone_loads.csv')
