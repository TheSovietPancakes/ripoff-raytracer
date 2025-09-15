#!/usr/bin/bash

# This script will create or clear an img/ directory,
# start the program, and then use ffmpeg to
# combine the frames into a neat little video.

./build.sh
mkdir -p img
rm img/*
./build/gputest
rm video.mp4 video.gif
ffmpeg -framerate 30 -i img/output_%d.bmp -c:v libx264 -pix_fmt yuv420p video.mp4
# Generate a GIF from the mp4. Output is 512x512, 30fps, looped.
ffmpeg -i video.mp4 -vf "fps=30,scale=512:-1:flags=lanczos" -loop 0 video.gif