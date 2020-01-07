#!/usr/bin/env python
#
# talker.py
# Class to faciliate publishing valid poses from a camera for posenet_wrapper ROS package.
# Author: Matthew Yu
# Last Modified: 1/7/20
# Organization: UT Austin SIMLab

DIST = 'melodic' # replace based on current ROS distribution (melodic, etc)


import rospy
from posenet_wrapper.msg import Pose

import sys
sys.path.remove('/opt/ros/' + DIST + '/lib/python2.7/dist-packages')
# find all non ros modules first
import importlib
import torch
import cv2
import time
import argparse
import numpy as np
import posenet
# import libraries for realsense D435
import pyrealsense2 as rs
sys.path.append('/opt/ros/' + DIST + '/lib/python2.7/dist-packages')

FREQ = 5

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--realsense', type=int, default=0)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=480)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()

def talker():
    # setup model and camera
    model = posenet.load_model(101)
    output_stride = model.output_stride

    if args.realsense is 1:
        # setup realsense camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        # start up the publisher and node
        pub = rospy.Publisher('posenet', Pose, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(FREQ) # 10hz
        start = time.time()
        frame_count = 0
        try:
            while not rospy.is_shutdown():
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                #480 by 640
                input_image, display_image, output_scale = posenet.read_cap(
                    color_image, realsense=True, scale_factor=0.7125, output_stride=output_stride)

                with torch.no_grad():
                    input_image = torch.Tensor(input_image)
                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                        heatmaps_result.squeeze(0),
                        offsets_result.squeeze(0),
                        displacement_fwd_result.squeeze(0),
                        displacement_bwd_result.squeeze(0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)

                    keypoint_coords *= output_scale

                # TODO this isn't particularly fast, use GL for drawing and display someday...
                overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)

                # Show images
                cv2.namedWindow('posenet', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('posenet', overlay_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # filter out empty frames with no identifiable poses
                if(pose_scores[0] != 0.0):
                    data = Pose()
                    data.header.frame_id    = str(frame_count)
                    data.header.stamp.secs  = int(time.time() - start)
                    data.pose_scores        = pose_scores[0:1].tolist()
                    data.keypoint_scores    = keypoint_scores[0].tolist()
                    data.keypoint_coords    = keypoint_coords[0].flatten().tolist()
                    rospy.loginfo(data)
                    pub.publish(data)

                rate.sleep()
                frame_count += 1

        finally:
            pipeline.stop()
            pub.publish("Shutting down.")

    else:
        # setup integrated camera
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # start up the publisher and node
        pub = rospy.Publisher('posenet', Pose, queue_size=10)
        rospy.init_node('talker', anonymous=True)
        rate = rospy.Rate(FREQ) # 10hz
        start = time.time()
        frame_count = 0
        try:
            while not rospy.is_shutdown():
                #480 by 640
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=0.7125, output_stride=output_stride)

                with torch.no_grad():
                    input_image = torch.Tensor(input_image)
                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                        heatmaps_result.squeeze(0),
                        offsets_result.squeeze(0),
                        displacement_fwd_result.squeeze(0),
                        displacement_bwd_result.squeeze(0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)

                    keypoint_coords *= output_scale

                overlay_image = posenet.draw_skel_and_kp(
                    display_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.15, min_part_score=0.1)

                # Show images
                cv2.namedWindow('posenet', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('posenet', overlay_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # filter out empty frames with no identifiable poses
                if(pose_scores[0] != 0.0):
                    data = Pose()
                    data.header.frame_id    = str(frame_count)
                    data.header.stamp.secs  = int(time.time() - start)
                    data.pose_scores        = pose_scores[0:1].tolist()
                    data.keypoint_scores    = keypoint_scores[0].tolist()
                    data.keypoint_coords    = keypoint_coords[0].flatten().tolist()
                    rospy.loginfo(data)
                    pub.publish(data)

                rate.sleep()
                frame_count += 1

        finally:
            pub.publish("Shutting down.")


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
