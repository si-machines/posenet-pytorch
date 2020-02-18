#!/usr/bin/env python
#
# Classifier.py
# Author: Michelle Wen
# Last Modified: 1/20/2020
# Organization: UT Austin SIMLab
DIST = 'kinetic' # replace based on current ROS distribution (melodic, etc)
USER = 'michellewen'
import rospy
from posenet_wrapper.msg import Pose
from std_msgs.msg import String

import sys
import os

sys.path.remove('/opt/ros/' + DIST + '/lib/python2.7/dist-packages')
#sys.path.append("/home/${USER}/anaconda3/lib/python3.7/site-packages")
import glob
import argparse
import numpy as np
import math
import statistics
import posenet
import cv2
#sys.path.remove("/home/${USER}/anaconda3/lib/python3.7/site-packages")
sys.path.append('/opt/ros/' + DIST + '/lib/python2.7/dist-packages')

PATH = "/home/" + USER + "/catkin_ws/src/posenet_wrapper/frame_data_example"
FREQ = 5

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0)  # default multishot
args = parser.parse_args()


class Classifier(object):
    data_points = []
    library = []

    def __init__(self):
        rospy.init_node('classifier', anonymous=True)
        self.load_data(PATH)
        self.publisher = rospy.Publisher('classifications', String, queue_size=10)
        self.subscriber = rospy.Subscriber('posenet', Pose, self.cb_classify)

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

        # center the keypoints
        adjusted_points = posenet.center_of_gravity(self.data_points[3])

        # nearest neighbors
        classified_pose = self.knn(adjusted_points)

        # publish result to a topic
        print("Classified pose: " + classified_pose)
        # rospy.loginfo(classified_pose)
        self.publisher.publish(classified_pose)

        """
        raw pose is a list of data points
        self.data_points =
        [string,     # id
        list,        # pose_scores
        list,        # keypoint_scores
        list]        # keypoint_coords

        self.library is a list of labeled poses
        self.library[i] =
        [integer,   # frame ID
        tuple,      # pose score ()
        tuple,      # keypoint scores ()    !!!
        tuple,      # keypoint coordinates ()   !!!
        string]     # label
        """

    def knn(self, coord_points):
        """
        nearest neighbors implementation between captured pose and self.library
        """
        # raw_coords_list = self.data_points[3]
        raw_coords_list = coord_points
        dist_table = []
        k = 2
        k_nearest_labels = []

        for labeled_pose in self.library:
            total_distance = 0  # centers keypoints after grabbing original keypoints from frame_data_example
            # labeled_coords_list = posenet.center_of_gravity(labeled_pose[3])
            labeled_coords_list = labeled_pose[3]   # assuming poses in library are centered
            for rawX, rawY, labeledX, labeledY, labeled_keyscore in zip(raw_coords_list[0::2],
            raw_coords_list[1::2], labeled_coords_list[0::2], labeled_coords_list[1::2], labeled_pose[2]):
                # distance between 2 coordinates
                distance = math.sqrt(((rawX - labeledX) ** 2) + ((rawY - labeledY) ** 2))
                distance *= labeled_keyscore
                total_distance += distance

            dist_table.append((total_distance, labeled_pose[4]))

        dist_table.sort()
        k_nearest_distances = dist_table[:k]

        for pose in k_nearest_distances:
            k_nearest_labels.append(pose[1])

        if k_nearest_labels == []:
            classified_pose = "No labeled data"
        else:
            try:
                classified_pose = statistics.mode(k_nearest_labels)
            except:
                classified_pose = k_nearest_labels[0]

        # classified_pose = "pose2"
        return classified_pose

    def load_data(self, path_name):
        """
        - dir_name is directory to all frame files
        - each frame is represented as a list of values [integer, tuple, tuple, tuple, string]
        """
        for file_name in os.listdir(path_name):
            file_path = path_name + '/' + file_name
            try:
                frame = np.load(file_path, allow_pickle=True)
                self.library.append(frame)
            except:
                pass

        """
        # debugging
        file_path = PATH + '/' + "2019-12-02_s1385_f4157"
        frame = np.load(file_path, allow_pickle=True)
        self.library.append(frame)
        """

if __name__ == '__main__':
    classifier = Classifier()
