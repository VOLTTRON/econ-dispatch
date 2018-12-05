from econ_dispatch.component_models import ComponentBase
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

EXPECTED_PARAMETERS = set([
                            "boilers",
                            "generators",
                            "pmax",
                            "import_peak_on_startup",
                            "parasite",
                            "demand_charge",
                            "minimum_import"
                        ])

class Component(ComponentBase):
    def __init__(self,
                 boilers=[],
                 generators=[],
                 pmax=0,
                 import_peak_on_startup=0,
                 parasite=0,
                 demand_charge=0,
                 minimum_import=0,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        self.boilers = boilers
        self.parameters["boilers"] = self.boilers
        self.generators = generators
        self.parameters["generators"] = self.generators
        self.pmax = pmax
        self.parameters['pmax'] = self.pmax
        self.import_peak_on_startup = import_peak_on_startup
        self.parameters['import_peak_on_startup'] = self.import_peak_on_startup
        self.parasite = parasite
        self.parameters['parasite'] = self.parasite
        self.demand_charge = demand_charge
        self.parameters['demand_charge'] = self.demand_charge
        self.minimum_import = minimum_import
        self.parameters['minimum_import'] = self.minimum_import

    def get_output_metadata(self):
        return []

    def get_input_metadata(self):
        return []

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        return {}