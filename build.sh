#!/bin/bash

docker build --pull --rm \
    "." \
    -t autonomous_f1tenth:latest
