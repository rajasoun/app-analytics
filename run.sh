#!/usr/bin/env sh

make env
. .env/bin/activate
python3 notebook/demand_forecast_by_ga.py
