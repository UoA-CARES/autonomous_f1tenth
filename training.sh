#!/bin/bash

cd /ws/src/f1tenth 
git fetch
git checkout main
colcon build
cd /ws
git fetch
git checkout P60003
git pull
colcon build
ros2 launch reinforcement_learning car_track_1.launch.py