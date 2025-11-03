# raytracer project

A simple raytracer written in C++, using OpenCL for sending jobs to the GPU.

At the moment, this project is not particularly optimized, other than one feature...
If you happen to own multiple GPUs (assuming you have your OpenCL drivers installed :P), this program can automatically
detect all compatible devices and split the load of a render amongst them! Work is based on a queue, so even if you have
one very powerful and one very weak GPU, the work will still be split fairly, where the powerful GPU can perform more tasks
and thus the minimal possible time is reached every time.

Although, you likely only have one good GPU, and not two or more, but that is OK! This project will still work perfectly fine
for you. :D

## Building and Installing and Whatever

Obviously, it depends on what platform you're going to build to. So, refer to your OS below:

### Windows

0. Download this repository onto your computer. You can either click the green "Code" button on GitHub (top-left of this page) or you can use git on your local machine.

1. Open a new Powershell window as Administrator.
2. Go to the directory where this repository has been cloned using the `cd` command followed by the path (e.g. `cd C:\path\to\repo`).
3. From within that folder, run this nice build script I made for by doing `.\windows.ps1`.
3.1. ONLY run this script if you have not installed Chocolatey, CMake, Make, and OpenCL. Otherwise, you can skip this altogether.
4. Wait a good couple seconds. It will check and install Chocolatey, CMake, Make, and check for OpenCL.
5. After you see "Hit enter to continue," press Enter.
6. Now, you should close the Powershell window and open a new CMD/Powershell window (it doesn't need to be admin this time) to refresh the PATH, if anything was installed.
7. Navigate to the repository folder again using `cd`.
8. Run these commands:

```sh
mkdir build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"
cmake --build build
```

9. You're done! You can close the terminal window. The executable will be in the `build` folder.

### Linux

If you use Linux, you have probably built CMake projects thousands of times. If you haven't though, then this one is no different than those :P. Still, if needed, here are the steps:

Depending on the distro you use, the way you install dependencies may vary. Here is a rough outline for most popular distros:

1. Open a new terminal window.
2. Install dependencies.
   For Debian-based distros (Ubuntu, Pop!_OS, Debian, etc.):

```sh
sudo apt update && sudo apt install cmake ocl-icd-opencl-dev opencl-headers build-essential git
```

3. Download this repository if you haven't already. You can choose to download a ZIP archive of the latest push or clone from CLI using git.
4. From within the terminal, navigate to the folder where this repository is located using `cd /path/to/repo`.
5. You should be good-to-go for building. You can run the nice build script I made for Linux by doing `./build.sh` to automatically create a `build` folder, run CMake, and build the project for Release.
6. You're done! You can close the window. The executable will be in the `build` folder.

### The Lazy Way Out

There is none. You have to download your own damn 3D models, build your own code. Also, don't you think it's a good exercise to learn how to build CMake projects? They're very popular.

## Usage

Run the executable. It takes in no arguments, so you can double click it. It will open a new terminal with a progress indicator and output a BMP file in the CWD it was run from called `output.bmp`.
