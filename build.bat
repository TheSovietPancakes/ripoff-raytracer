@echo off

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
cmake --build build