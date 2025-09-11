#!/usr/bin/bash

# if an argument is passed, build in debug
DEBUG=0
if [ "$1" == "debug" ]; then
    DEBUG=1
fi

if [ $DEBUG -eq 1 ]; then
    echo "Debug mode"
    mkdir -p debug
    cmake -S . -B debug -DCMAKE_BUILD_TYPE=Debug -G Ninja
    cmake --build debug
    exit 0
elif [ $DEBUG -eq 0 ]; then
    echo "Release mode"
    mkdir -p build
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
    cmake --build build
    exit 0
fi