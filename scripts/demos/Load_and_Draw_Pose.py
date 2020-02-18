#!/usr/bin/env python
#
# Load_and_Draw_Pose.py
# Example to demonstrate pose loading and display from pickled file.
# Author: Matthew Yu
# Last Modified: 12/11/19
# Organization: UT Austin SIMLab
import Recorder

FILE_NAME = "2019-12-11_s94_f474"
LABEL = "standing"

r = Recorder.Recorder()

frame = r.load_data("../frame_data_example/"+FILE_NAME)

# draw pose
r.draw_pose_2(frame)

# relabel?
# r.label_self(FILE_NAME, LABEL)
