### Install

A suitable Python 3.x environment with a recent version of PyTorch is required. Development and testing was done with Python 3.7.1 and PyTorch 1.0 w/ CUDA10 from Conda.

If you want to use the webcam demo, a pip version of opencv (`pip install python-opencv=3.4.5.20`) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. The python bindings for OpenCV 4.0 currently have a broken impl of drawKeypoints so please force install a 3.4.x version.

A fresh conda Python 3.6/3.7 environment with the following installs should suffice:
```
conda install -c pytorch pytorch cudatoolkit
pip install requests opencv-python==3.4.5.20
```

# SIMLAB INSTRUCTIONS
in order to use on Moe, you must first make posenet point to the correct python distribution:
`export PYTHONPATH="/home/moe/anaconda3/lib/python3.7/site-packages:/home/moe/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages"`

# Basic Setup
* edit PYTHONPATH - `export PYTHONPATH="/home/{USER}/anaconda3/lib/python3.7/site-packages:/home/{USER}/catkin_ws/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages"`
* build the entire catkin_ws - `catkin_make`
* source workspace - `source ~/catkin_ws/devel/setup.bash`
* startup roscore - `roscore`
* startup talker.py - `rosrun posenet_wrapper talker.py`
* startup listener.py - `rosrun posenet_wrapper listener.py`


# PC Installation:
Install dependencies:
1. `sudo apt install python-pip`
2. `sudo apt install python3-pip`
3. `pip install torch torchvision`
4. `sudo apt install python3-opencv`
5. `pip install pyrealsense2`

Test basic functionality: webcam_demo.py

1. `cd /posenet-pytorch/scripts/`
2. `mv demos/webcam_demo.py .`
3. `python webcam_demo.py \[--cam_width=1280\] \[--cam_height=720\]` 
   1.  Adjust default args to appropriate camera resolution.
   2.  Assumes the camera of the computer is id=0.
   3.  To run on the realsense instead of a default camera, add the flag \[--realsense=1\]

Setup ROS (Kinetic/Melodic)
1. [Follow setup page](http://wiki.ros.org/melodic/Installation/Ubuntu)
2. `mkdir -p ~/catkin_ws/src`
3. `cd ~/catkin_ws/`
4. `catkin_make`
   1. Make sure that the basic setup compiles successfully.
5. `mv ~/PATH_TO_REPO/posenet-pytorch ./src/`
6. Follow **Basic Setup** instructions.