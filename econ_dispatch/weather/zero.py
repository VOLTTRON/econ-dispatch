from datetime import timedelta
import logging
_log = logging.getLogger(__name__)

from econ_dispatch import utils
from econ_dispatch.forecast_models import ForecastModelBase

time_step = timedelta(hours=1)

class Weather(ForecastModelBase):
    def __init__(self, hours_forecast=24, **kwargs):
        super(Weather, self).__init__(**kwargs)
        self.hours_forecast = hours_forecast

    def derive_variables(self, now, independent_variable_values={}):
        """Get the predicted load values based on the independent variables."""
        pass


    def get_weather_forecast(self, now):
        return [dict(timestamp = now + n*time_step) for n in xrange(self.hours_forecast)]
