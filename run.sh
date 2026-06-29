#!/bin/bash

while getopts p: flag
do
    case "${flag}" in
        p) GZ_PARTITION=${OPTARG};;
    esac
done

xhost +

docker run --rm -it \
    --network host \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_COMPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    -e renderingEngine=ogre2-2 \
    -e DISPLAY \
    -e GZ_PARTITION=$GZ_PARTITION \
    -e ROS_DOMAIN_ID=$GZ_PARTITION \
    -v "$PWD/data:/ws/data" \
    -v "$PWD/models:/ws/models" \
    -v "$PWD/figures:/ws/figures" \
    -v "$PWD/autonomous_bringup:/ws/autonomous_bringup" \
    -v "$PWD/f1tenth_controllers:/ws/f1tenth_controllers" \
    -v "$PWD/f1tenth_interfaces:/ws/f1tenth_interfaces" \
    -v "$PWD/f1tenth_environments:/ws/f1tenth_environments" \
    -v "$PWD/f1tenth_gazebo:/ws/f1tenth_gazebo" \
    -v "$PWD/f1tenth_recorders:/ws/f1tenth_recorders" \
    autonomous_f1tenth:latest \
    bash
 