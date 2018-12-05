from econ_dispatch.forecast_models import HistoryModelBase
import pandas as pd
import numpy as np

class Model(HistoryModelBase):
    def train(self, training_data):
        super(Model, self).train(training_data)

        # Monday-Friday, 7:00-20:00, inclusive
        peak_hours = [True if (i.weekday() in np.arange(0,6)) and (i.hour in np.arange(7,21)) else False for i in self.historical_data['timestamp']]
        self.historical_data["peak_hours"] = pd.Series(peak_hours)

        try:
            cpi = self.historical_data['cpi']
        except KeyError:
            cpi = 1
        gas_price = self.historical_data['gas_price']
        self.historical_data["college_electric_price"] = 0.08158 + 0.00998*cpi + 0.0073*(gas_price - 11.22)
        self.historical_data["hospital_electric_price"] = 0.03214 + 0.01337*cpi + 0.0029*(gas_price - 11.22)
        self.historical_data["home_electric_price"] = 0.10537 - 0.00293*cpi + 0.0094*(gas_price - 11.22)

        facilities = ['college', 'hospital', 'home']
        self.historical_data = self.historical_data[['timestamp', 'peak_hours', 'export_price'] 
                + ["{}_electric_price".format(f) for f in ['college', 'hospital', 'home']] 
                + ["{}_import_price".format(f) for f in ['college', 'hospital', 'home']]]
