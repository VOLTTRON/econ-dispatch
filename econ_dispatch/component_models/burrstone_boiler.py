from econ_dispatch.component_models import ComponentBase
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

EXPECTED_PARAMETERS = set(["fundata"])

class Component(ComponentBase):
    def __init__(self, **kwargs):
        super(Component, self).__init__(**kwargs)

        self.output = 0
        self.command_history = [0] * 24

    def get_output_metadata(self):
        return [u"steam"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        set_point = component_loads["boiler_gas_{}_0".format(self.name)]
        
        run_boiler = set_point > 0.0

        # self.output = self.max_output if run_boiler else 0
        self.output = set_point if run_boiler else 0
        self.parameters["output"] = self.output

        # self.command_history = self.command_history[1:] + [int(set_point>0)]
        # self.parameters["command_history"] = self.command_history[:]

        return {"command": int(run_boiler)}