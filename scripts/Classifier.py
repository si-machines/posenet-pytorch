#!/usr/bin/env python
#
# Classifier.py
# Author: Michelle Wen
# Last Modified: 12/10/19
# Organization: UT Austin SIMLab

import rospy
from posenet_wrapper.msg import Pose
from std_msgs.msg import String

import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import glob
import argparse
import numpy as np
import posenet
import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

PATH = "./src/posenet_wrapper/frame_data_example/"
FREQ = 5

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)  # default multishot
args = parser.parse_args()


class Classifier(object):
    data_points = []
    library = []

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)
        self.load_data(glob.glob(PATH))
        self.publisher = rospy.Publisher('classifications', String, queue_size=10)
        self.subscriber = rospy.Subscriber('chatter', Pose, self.cb_classify)

        rospy.spin()

    def cb_classify(self, data):
        """
        capture frame data, find k nearest neighbors, publish pose classification
        """
        self.data_points = [
            data.header.frame_id,
            data.pose_scores,
            data.keypoint_scores,
            data.keypoint_coords
        ]

        # nearest neighbors
        classified_pose = self.knn()

        # publish result to a topic
        rospy.loginfo(classified_pose)
        self.publisher.publish(classified_pose)

    def knn(self):
        # nearest neighbors implementation between captured pose and self.library
        classified_pose = "pose1"
        return classified_pose

    def load_data(self, dir_name):
        """
        - dir_name is directory to all frame files
        - each frame is represented as a list of values [integer, tuple, tuple, tuple, string]
        """
        for file_name in dir_name:
            self.library.append(np.load(file_name, allow_pickle=True))


if __name__ == '__main__':
    classifier = Classifier()
