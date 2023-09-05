#!/bin/bash

cd ~/autonomous_f1tenth/

if [ -z "$1" ]; then
  echo "Please enter the GZ_PARTITION number"
  echo "Usage: ./4openGZ.sh <GZ_PARTITION>"
  return
fi

export GZ_PARTITION=$1
gz sim -g
