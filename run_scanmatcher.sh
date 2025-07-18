#!/bin/bash
PYTHON_VENV=~/Applications/venv2
source $PYTHON_VENV/bin/activate
export PYTHONPATH=$PYTHON_VENV/lib/python3.8/site-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:
python run_scanmatcher.py
