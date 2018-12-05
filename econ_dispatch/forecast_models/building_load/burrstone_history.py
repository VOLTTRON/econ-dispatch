from econ_dispatch.forecast_models import HistoryModelBase

class Model(HistoryModelBase):
    def train(self, training_data):
        super(Model, self).train(training_data)

        self.historical_data = self.historical_data[['timestamp', 'heat_load', 'steam_load', 'college_elec_load', 'hospital_elec_load', 'home_elec_load']]
        