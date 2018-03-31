#!/usr/bin/env bash
python main.py new_agent2.config --start-time "2017-07-01 00:00" --end-time "2017-07-02 00:00" --input-data "battery_data.csv" --input-data-time-column "Date/Time" --output-data "command_output.csv"
