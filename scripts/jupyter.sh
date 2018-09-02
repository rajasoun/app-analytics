#!/usr/bin/env sh

nohup jupyter notebook --port=8888 --no-browser > log/jupyter.log &
