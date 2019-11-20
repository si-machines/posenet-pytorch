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
import numpy as np
import posenet
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

class Recorder(object):
    """
    The recorder class sets up a ROS subscriber node to listen to a specific publisher.
    """
    data_points = []
    empty_canvas = np.zeros((480,640))
    idx = 0

    def __init__(self):
        # by default, the Recorder is not recording.
        self.recording = False

        rospy.init_node('listener', anonymous=True)
        self.subscriber = rospy.Subscriber('chatter', Pose, self.cb_pose)
        print(self.subscriber)

    def cb_pose(self, data):
        if not self.recording:
            return

        # rospy.loginfo('F_ID: %s\n', data.header.frame_id)
        # print("P_Scores: ", data.pose_scores)
        # print("K_Scores: ", data.keypoint_scores)
        # print("K_Coords: ", data.keypoint_coords)

        Recorder.data_points.append(data)
        recorder.draw_pose()
        Recorder.idx = Recorder.idx + 1

    def enable_cb(self, enable):
        self.recording = True if (enable == True) else False
        if self.recording:
            print("Recording enabled.")
        else:
            print("Recording disabled.")

    # primitive data save function. TODO: convert pose data into JSON/CSV
    def save_data(self, file_name):
        np.savetxt(file_name, Recorder.data_points, fmt="%s")
        print("Data saved to", file_name)

    def draw_pose(self):
        # print(Recorder.data_points[Recorder.idx].pose_scores)
        # print(Recorder.idx, "printing ii: ")
        # for ii, score in enumerate(Recorder.data_points[Recorder.idx].pose_scores):
        #     print(ii)
        #     print(Recorder.data_points[Recorder.idx].keypoint_scores[ii:])
        # print("End iteration")
        # print(Recorder.data_points[Recorder.idx].keypoint_scores)
        # print(Recorder.data_points[Recorder.idx].keypoint_coords)

        overlay_image = posenet.draw_skel_and_kp(
            Recorder.empty_canvas,
            Recorder.data_points[Recorder.idx].pose_scores,
            Recorder.data_points[Recorder.idx].keypoint_scores,
            Recorder.data_points[Recorder.idx].keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        # Show images
        cv2.namedWindow('posenet', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('posenet', overlay_image)

    def main_seq(self, prep_time, recording_time, file_name="data.txt"):
        # startup sequence: press enter to start a n second sequence
        start = input("Press enter to begin.")
        for sec in range(0, prep_time):
            print("Starting sequence in %3d seconds." % (prep_time-sec))
            time.sleep(1)

        recorder.enable_cb(True)
        print("Recording. %3d second listening period begin." % recording_time)
        time.sleep(recording_time)
        recorder.enable_cb(False)
        print("End recording.")

        recorder.save_data(file_name)
        end = input("Press enter to exit.")


if __name__ == '__main__':
    recorder = Recorder()
    recorder.main_seq(3, 5, "pose_data.txt")
