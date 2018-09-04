#!/usr/bin/env sh

virtualenv --python=python3 .env
source .env/bin/activate
pip3 install -r requirements.txt
scripts/jupyter.sh
python3 notebook/demand_forecast_by_ga.py
