.. _DataPublisher:

=============
DataPublisher
=============

This is a simple agent that plays back data either from the config
store or a CSV to the configured topic. It can also provide basic
emulation of the actuator agent for testing agents that expect to
be able to set points on a device.

Installation notes
------------------

In order to simulate the actuator you must install the agent
with the VIP identity of `platform.actuator`.

Configuration
-------------

::

    {
        # basetopic can be devices, analysis, or custom base topic
        "basepath": "devices/PNNL/ISB1",
        "use_timestamp": true,

        "max_data_frequency": 900, #Only used if timestamp in input file is used.

        # The meta data published with the device data is generated
        # by matching point names to the unittype_map.
        "unittype_map": {
            ".*Temperature": "Farenheit",
            ".*SetPoint": "Farenheit",
            "OutdoorDamperSignal": "On/Off",
            "SupplyFanStatus": "On/Off",
            "CoolingCall": "On/Off",
            "SupplyFanSpeed": "RPM",
            "Damper*.": "On/Off",
            "Heating*.": "On/Off",
            "DuctStatic*.": "On/Off"
        },
        # Path to input CSV file.
        # May also be a list of records or reference to a CSV file in the config store.
        # Large CSV files should be referenced by file name and not
        # stored in the config store.
        "input_data": "econ_test2.csv",
        # Publish interval in seconds
        "publish_interval": 1,

        # Tell the playback to maintain the location in the file in the config store.
        # Playback will be resumed from this point
        # at agent startup even if this setting is changed to false before restarting.
        # Saves the current line in line_marker in the DataPublishers's config store
        # as plain text.
        # default false
        "remember_playback": true,

        # Start playback from 0 even though remember playback may be set.
        # default false
        "reset_playback": false,

        # Repeat data from the start if this flag is true.
        # Useful for data that does not include a timestamp and is played back in realtime.
        "replay_data": false
    }
