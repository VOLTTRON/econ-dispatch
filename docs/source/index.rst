###################################
Documentation for Economic Dispatch
###################################

Dispatch Software for Combined Cooling, Heating and Power Systems

A multi-purpose, open-source control algorithm that ensures real-time optimal
operation, increases electric grid reliability, and leads to the goal of a
clean, efficient, reliable and affordable next generation building-integrated
combined cooling, heating and power system. The CHP system could includ
conventional heating, ventilation, air conditioning (HVAC) systems,
distributed generation (DG), local storage (both thermal and electric) and
local solar photovoltaic (PV) systems.

********
Features
********

- Build and train models of CHP components
- Query utility APIs
- Predict building loads
- Determine optimal component operations strategy
- Dispatch commands to the CHP system

********
Platform
********
|VOLTTRON Tagline|

Economic Dispatch is built on the `VOLTTRON <https://volttron.readthedocs.io/en/develop>`_\ :sup:`TM`
platform.

VOLTTRON\ :sup:`TM` is an open-source platform for distributed sensing and
control. The platform provides services for collecting and storing data from
buildings and devices and provides an environment for developing applications
that interact with that data.

Economic Dispatch can stand alone: it requires no other agents to operate.
However, certain capabilities will require  the use of other agents. See the
VOLTTRON docs for details:

- `Historian <https://volttron.readthedocs.io/en/develop/core_services/historians/index.html>`_:
  stores record of activity for component re-training
- `Driver <https://volttron.readthedocs.io/en/develop/core_services/drivers/index.html>`_:
  interfaces with components

************
Installation
************

Please see the VOLTTRON `documentation <https://volttron.readthedocs.io/en/develop/setup/index.html>`_ for
VOLTTRON installation instructions.

Once VOLTTRON is installed, install Economic Dispatch like any VOLTTRON agent:

.. code-block:: bash

  python ${VOLTTRONDIR}/scripts/install-agent.py -s ${ECONDISPATCHDIR}/econ_dispatch/ -c ${ECONDISPATCHDIR}/config -t econ_dispatch

**********
Next Steps
**********

- Read the :ref:`overview` for a high-level description of 
  how the agent works.
- Read the :ref:`walkthrough` to get started with the included example models.
- Read the full API documentation to learn how to develop your own component, 
  forecast, and optimization models: :ref:`modindex`

**********
Contribute
**********

How to contribute back:

- Join the :doc:`VOLTTRON <volttron:community_resources/index>` community! 
- Issue Tracker: https://github.com/VOLTTRON/econ-dispatch/issues
- Source Code: https://github.com/VOLTTRON/econ-dispatch

********
Contents
********
.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Overview <specs/overview>
   Curve Fitting <specs/curve-fitting>
   Walkthrough <specs/walkthrough>
   API <_api/modules>

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |VOLTTRON Tagline|
  image:: ./specs/images/VOLLTRON_Logo_Black_Horizontal_with_Tagline.png
  :width: 300px
