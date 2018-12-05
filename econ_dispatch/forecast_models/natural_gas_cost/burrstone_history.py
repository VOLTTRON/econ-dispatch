from econ_dispatch.forecast_models import HistoryModelBase

params = {
    "st_loss": 0.2,
    "st_hfg": 1019.4,
    "lhv": 905,
    "hhv": 905/0.9,
    "eta_boiler_real": 0.62,
    "q_kw": 0.005,
}

class Model(HistoryModelBase):
    def train(self, training_data):
        super(Model, self).train(training_data)
        
        try:
            cpi = self.historical_data['cpi']
        except KeyError:
            cpi = 1
        
        self.historical_data["thermal_price"] = 9.421 + 3.918*cpi + 0.8396*(self.historical_data['gas_price'] - 11.22)
        self.historical_data["heat_price"] = self.historical_data["thermal_price"]/(1-params['st_loss'])
        self.historical_data["steam_price"] = self.historical_data["thermal_price"]*params['st_hfg']/1e6

        self.historical_data['gas_price'] = self.historical_data['gas_price']*params['hhv']/1e6

        self.historical_data = self.historical_data[['timestamp', 'gas_price', 'heat_price', 'steam_price']]