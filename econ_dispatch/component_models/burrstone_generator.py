from econ_dispatch.component_models import ComponentBase
from econ_dispatch import utils
import logging

_log = logging.getLogger(__name__)

EXPECTED_PARAMETERS = set([
                            "fundata",
                            "min_on",
                            "min_off",
                            "capacity",
                            "maint_cost",
                            "command_history"
                          ])

class Component(ComponentBase):
    def __init__(self,
                 min_on=0,
                 min_off=0,
                 capacity=0,
                 maint_cost=0,
                 **kwargs):
        super(Component, self).__init__(**kwargs)

        self.min_on = min_on
        self.parameters['min_on'] = self.min_on
        self.min_off = min_off
        self.parameters['min_off'] = self.min_off
        self.capacity = capacity
        self.parameters['capacity'] = self.capacity
        self.maint_cost = maint_cost
        self.parameters['maint_cost'] = self.maint_cost

        self.output = 0
        self.command_history = [0] * 24
        self.parameters['command_history'] = self.command_history

    def get_output_metadata(self):
        return [u"electricity", u"heated_water", u"steam"]

    def get_input_metadata(self):
        return [u"natural_gas"]

    def validate_parameters(self):
        parameters = [self.parameters.get(x) for x in EXPECTED_PARAMETERS]
        return None not in parameters

    def get_mapped_commands(self, component_loads):
        set_point = component_loads["generator_elec_{}_0".format(self.name)]

        self.output = set_point
        self.parameters["output"] = self.output

        # Do not update command history while testing
        # self.command_history = self.command_history[1:] + [int(set_point>0)]
        self.parameters["command_history"] = self.command_history[:]

        return {"command": int(set_point>0)}
