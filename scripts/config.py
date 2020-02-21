#!/usr/bin/env python
#
# config.py
# Stores important global variables for the posenet_wrapper ROS package.
# Author: Matthew Yu
# Last Modified: 2/21/20
# Organization: UT Austin SIMLab
DIST = 'melodic' # replace based on current ROS distribution (melodic, etc)
FREQ = 5         # frequency that the camera grabs frames
SCREEN_WIDTH= 480   # default expectation that Recorder will project onto a canvas
SCREEN_HEIGHT = 640 # default expectation that Recorder will project onto a canvas
PATH = "./src/posenet_wrapper/frame_data_example/"
