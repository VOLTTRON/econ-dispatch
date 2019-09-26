# Economic Dispatch

## Dispatch Software for Combined Cooling, Heating and Power Systems

A multi-purpose, open-source control algorithm that ensures real-time optimal
operation, increases electric grid reliability, and leads to the goal of a
clean, efficient, reliable and affordable next generation building-integrated
combined cooling, heating and power system. The CHP system could includ
conventional heating, ventilation, air conditioning (HVAC) systems,
distributed generation (DG), local storage (both thermal and electric) and
local solar photovoltaic (PV) systems.

## Features

- Build and train models of CHP components
- Query utility APIs
- Predict building loads
- Determine optimal component operations strategy
- Dispatch commands to the CHP system

## Platform
Economic Dispatch is built on the [VOLTTRON<sup>TM</sup>](https://volttron.readthedocs.io/en/develop>)
platform.

VOLTTRON<sup>TM</sup> is an open-source platform for distributed sensing and
control. The platform provides services for collecting and storing data from
buildings and devices and provides an environment for developing applications
that interact with that data.

Economic Dispatch can stand alone: it requires no other agents to operate.
However, certain capabilities will require  the use of other agents. See the
VOLTTRON docs for details:

- [Historian](https://volttron.readthedocs.io/en/develop/core_services/historians/index.html):
  stores record of activity for component re-training
- [Driver](https://volttron.readthedocs.io/en/develop/core_services/drivers/index.html):
  interfaces with components

## Installation

Please see the [VOLTTRON documentation](https://volttron.readthedocs.io/en/develop/setup/index.html)
for VOLTTRON installation instructions.

Once VOLTTRON is installed, install Economic Dispatch like any VOLTTRON agent:

```bash
python ${VOLTTRONDIR}/scripts/install-agent.py -s ${ECONDISPATCHDIR}/econ_dispatch/ -c ${ECONDISPATCHDIR}/config -t econ_dispatch
```

## Documentation

Documentation is not available pre-compiled at this time, but may be compiled using Sphinx 
by executing, e.g., `make html` in the `docs` directory. This will require having the Python 
packages `sphinx` and `sphix_rtd_theme` installed.
