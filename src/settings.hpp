#pragma once

// VISITORS BEWARE:
// This program is super inefficient. Don't trust it as a reference for anything.
// Additionally, this has caused GPU freezes and crashes on my Linux PC many times.
// (Windows seems to throttle the GPU better, so it doesn't happen as often for me there.)
//
// As a precaution, please save everything in every program everywhere before you run this.
// In case of emergency, this program is fully safe to kill, end, or interrupt (^C).
///////////////////////////////////////////////////////////////////////////////////////////

// Program settings

// The program renders once with the camera at the start position and saves to output.bmp, then exits. No window is opened.
// If commented or removed, a window is created and a movable (WASD+QE+arrows) camera is used instead.
#define RENDER_AND_GET_OUT
// Give the GPU a break (~250ms) between frames to avoid crashing.
// Additionally, every 10 frames, a preview.bmp image is outputted to show current render progress.
// (since it's better to find out you mesed up after 10 frames rather than after 1,000.)
#define RELAX_GPU
// The start position of the camera, if RENDER_AND_GET_OUT is defined.
float CAMERA_START_X = 0.0f;
float CAMERA_START_Y = 150.0f;
float CAMERA_START_Z = 250.0f;
float CAMERA_START_PITCH = 0.0f;
float CAMERA_START_YAW = 3.14f;
float CAMERA_START_ROLL = 0.0f;
// On each frame, the pRNG's seed is different, and all frames are averaged together to produce a less noisy image.
// Obviously, the higher this is, the longer yet higher quality the render will be.
#define FRAME_TOTAL 1
// Functionally equivalent to FRAME_TOTAL, as it averages together pixel color results
// however, the averaging happens on the GPU, and images are brighter and quicker.
unsigned int RAYS_PER_PIXEL = 300;
unsigned int MAX_BOUNCE_COUNT = 50; // Self-explanatory
// The resolution of the output image and size of the window, if a window is created.
unsigned int WIDTH = 3840;
unsigned int HEIGHT = 2160;
// These are the dimensions of an iPhone 16, the phone that I have lol
// unsigned int WIDTH = 1179;
// unsigned int HEIGHT = 2556;
// Each frame is split into tiles so that the GPU has a change to refresh
// the screen and avoid crashing. However, if your GPU is powerful enough,
// a potential bottleneck could occur in data transfer between CPU/GPU.
// Update with caution.
unsigned int TILE_SIZE = 1024;
// The path, absolute or relative (to the cwd), to the .obj file to load.
#define OBJECT_PATH "dragon.obj"
// How much space there is inside the Cornell box between the model and the walls
#define CORNELL_BREATHING_ROOM 100.0f
// How many frames of video to render.
// If 1, a single output.bmp is produced. Otherwise, multiple frames are outputted to the VIDEO_FRAME_OUTPUT_DIR
#define VIDEO_FRAME_COUNT 1
// Video output FPS. If VIDEO_FRAME_COUNT is 1, this is ignored.
// !- NOTE! As of RIGHT NOW, this value is ignored anyways! FFmpeg will
// have to be run manually by the user. (It would just be yet another
// requirement for the user to install, and I would just feel bad.)
#define VIDEO_FPS 60
// The directory to output video frames to. Must exist already.
#define VIDEO_FRAME_OUTPUT_DIR "img"