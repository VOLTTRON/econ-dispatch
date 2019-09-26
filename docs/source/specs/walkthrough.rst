.. _walkthrough:

###########
Walkthrough
###########

User inputs are entered through configuration files which detail the components 
in the system, outputs, and the solver setup for that system.

The config file has five sections: specifications for the Agent, the 
Optimizer, the Weather Forecast, th eother Forecast Models, and the 
Component Models. The following describes the configuration file options. 
Compare the descriptions here to the example configuration files in the 
`examples` directory.

1. Agent

   * optimization schedule
      when to run the optimizer, either as an integer 
      number of minutes, or as a `cron string 
      <https://github.com/VOLTTRON/volttron/blob/a449f70e32f73ff0136a838d0feddb928ede6298/volttron/platform/scheduling.py#L195>`_.
   * training schedule
      when to re-train the components from the Volttron historian
   * schedule start
         default is current time. Useful for offline simulations
   * schedule end
         default is never. Useful for offline simulations
   * simulation mode
         wait for other agents before conducting the next 
         optimization. Useful for co-simulation with, e.g., 
         `EnergyPlus <https://energyplus.net/>`_.
   * offline mode
         don't wait before conducting the next timestep optimization
   * make reservations
         whether to reserve the actuator before pushing commands
   * historian vip id
         Volttron interconnect protocol identity of the local historian
   * optimizer debug
         where to save a CSV readout of the full optimizatized decision
   * command debug
         where to save a CSV readout of published commands

2. Optimizer

   * name
         name of optimizer file in `optimizer` directory
   * time limit
         integer, how long to let the optimizer run before publishing the 
         solution in seconds. If convergence is reached before the time
         limit, the solution is published on convergence. If convergence is not
         reached before the time limit, the current solution values are
         published.
   * write lp
         whether to write out a text files detailing the linear 
         programming optimizer poblem that is solved. Useful for debugging.

3. Weather

   * type
         name of forecast model file in `weather` directory
   * initial training data
         optional CSV file containing historical data
   * settings
         model-specific settings, such as

      * steps forecast
         number of time steps in optimization window
      * timestep
         duration in hours of each time step

4. Forecast Models

   * type
         name of forecast model file in `forecast_models` directory
   * name
         name of the forecast used when passing values to the optimizer
   * initial training data
         optional CSV containing historical data
   * settings
         model-specific settings, such as preprocessor instructions
         for historical data

5. Component Models

   * type
         name of component model file in `component_models` directory
   * name
         name of the component used when passing values to the optimizer
   * initial training data
         optional CSV containing historical data
   * default parameters
         initial parameters to pass to the optimizer --
         may be updated by training
   * outputs
         mapping of output keys to Volttron platform topic name
   * settings
         model-specific settings, such as capacity and preprocessor
         instructions for training data
