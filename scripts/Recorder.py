#!/usr/bin/env python
#
# Recorder.py
# Class to faciliate listener functions for posenet_wrapper ROS package.
# Author: Matthew Yu
# Last Modified: 11/20/19
# Organization: UT Austin SIMLab

import rospy
from posenet_wrapper.msg import Pose

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import datetime
import numpy as np
import posenet
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

PATH = "./src/posenet_wrapper/frame_data_example/"

class Recorder(object):
    """
    The recorder class sets up a ROS subscriber node to listen to a specific publisher.
    """
    lock = False
    data_points = []
    empty_canvas = np.zeros((480,640))

    def __init__(self):
        # by default, the `Recorder` is not recording.
        self.recording = False

        rospy.init_node('listener', anonymous=True)
        self.subscriber = rospy.Subscriber('chatter', Pose, self.cb_pose)
        print(self.subscriber)

    def cb_pose(self, data):
        """
        catches frame data from a custom rosmsg published by talker.py.
        exits early if disabled.
        """
        if not self.recording:
            return

        self.lock = True
        # rospy.loginfo('F_ID: %s\n', data.header.frame_id)
        # print("P_Scores: ", data.pose_scores)
        # print("K_Scores: ", data.keypoint_scores)
        # print("K_Coords: ", data.keypoint_coords)

        # build data point
        label = ""
        self.data_points = [
            data.header.frame_id,
            data.pose_scores,
            data.keypoint_scores,
            data.keypoint_coords,
            label
        ]

        # display pose on screen
        # recorder.draw_pose()
        # prompt for label
        self.label_auto()
        # save frame
        self.save_data(str(datetime.date.today()) + "_s" + str(data.header.stamp.secs) + "_f" + data.header.frame_id, self.data_points)

        self.lock = False

    def enable_cb(self, enable):
        """
        enables recording of pose data from talker.py.
        """
        self.recording = True if (enable == True) else False
        if self.recording:
            print("Recording enabled.")
        # else:
        #     print("Recording disabled.")

    def save_data(self, file_name, data):
        """
        data save function into pickle format.
        """
        np.asarray(data).dump(PATH + file_name)
        print("Data saved to", file_name)

    def load_data(self, file_name):
        """
        prints the frame given a valid file name. The frame is represented as a list of values. [integer, tuple, tuple, tuple, string]
        """
        frame = np.load(file_name, allow_pickle=True)
        print(frame)
        return frame

    def draw_pose(self):
        """
        TODO:
        Uses the posenet draw_skel_and_kp function in order to reconstruct a pose onto a blank canvas.
        Currently has tuple issues with keypoint_scores and keypoint_coords.
        """
        # overlay_image = posenet.draw_skel_and_kp(
        #     self.empty_canvas,
        #     list(self.data_points[1]),    # pose scores
        #     [list(self.data_points[2])],    # keypoint scores
        #     [[list(self.data_points[3])]],    # keypoint coords
        #     min_pose_score=0.15, min_part_score=0.1)

        # Show images
        cv2.namedWindow('posenet', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('posenet', overlay_image)

    def label_self(self, file_name, label):
        """
        adds a label to an saved pose frame given its file_name.
        does not check for similar pose labels.
        """
        frame = self.load_data(file_name)
        frame[4] = label
        self.save_data(file_name, frame)

    def label_auto(self):
        """
        adds a label to the most recently taken pose frame.
        does not check for similar pose labels.
        """
        self.data_points[4] = input("Enter a pose label: ")
        print(self.data_points)

    def main_seq(self, prep_time, recording_time, file_name="data.txt"):
        # startup sequence: press enter to start a n second sequence
        start = input("Press enter to begin.")
        for sec in range(0, prep_time):
            print("Starting sequence in %3d seconds." % (prep_time-sec))
            time.sleep(1)

        # begin recording period.
        self.enable_cb(True)
        print("Recording. %3d second listening period begin." % recording_time)
        time.sleep(recording_time)

        # don't kill the remaining callbacks, if any, until all processes finish.
        while(self.lock):
            self.enable_cb(False)

        print("End recording.")
        end = input("Press enter to exit.")

if __name__ == '__main__':
    recorder = Recorder()
    recorder.main_seq(3, 5, "pose_data.txt")
