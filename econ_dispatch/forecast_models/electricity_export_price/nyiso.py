import datetime
from csv import DictReader
from cStringIO import StringIO
import logging

import requests
import pytz

from econ_dispatch.forecast_models import ForecastModelBase


_log = logging.getLogger(__name__)

get_url = 'http://mis.nyiso.com/public/csv/rtlbmp/{date}rtlbmp_gen.csv'

MAX_UPDATE_FREQUENCY = datetime.timedelta(hours=0.5)

class Model(ForecastModelBase):
    def __init__(self, ptid=24049, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.last_collection = None
        self.current_value = None
        self.ptid=ptid

    def derive_variables(self, now, independent_variable_values={}):
        """Get the predicted load values based on the independent variables."""
        eastern = pytz.timezone('America/New_York')
        self.update_value(now.astimezone(eastern))

        return {"electricity_export_price": self.current_value}

    def update_value(self, now):
        if (self.last_collection is not None and
        (now - self.last_collection) < MAX_UPDATE_FREQUENCY):
            return

        today = now.date()

        r = requests.get(get_url.format(date=today.strftime("%Y%m%d")))
        try:
            r.raise_for_status()
        except StandardError as e:
            _log.warn(repr(e))
            return

        csv_data = StringIO(r.text)

        csv_reader = DictReader(csv_data)

        last_valid_row = None

        for row in csv_reader:
            if int(row["PTID"]) == self.ptid:
                last_valid_row = row

        rate = float(last_valid_row['LBMP ($/MWHr)'])

        self.current_value = rate

        self.last_collection = now