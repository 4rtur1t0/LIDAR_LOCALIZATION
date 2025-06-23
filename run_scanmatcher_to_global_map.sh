#!/bin/bash
source ~/Applications/venv2/bin/activate
export PYTHONPATH=~/Applications/venv2/lib/python3.8/site-packages:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:
python run_scanmatcher_to_global_map.py
