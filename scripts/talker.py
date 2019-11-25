#!/usr/bin/env python
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

import rospy
from posenet_wrapper.msg import Pose

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
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
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

def talker():
    # setup model and camera
    model = posenet.load_model(101)
    output_stride = model.output_stride

    # setup camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # start up the publisher and node
    pub = rospy.Publisher('chatter', Pose, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(3) # 10hz
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
                data.pose_scores        = pose_scores[0].flatten().tolist()
                data.keypoint_scores    = keypoint_scores[0].flatten().tolist()
                data.keypoint_coords    = keypoint_coords[0].flatten().tolist()
                rospy.loginfo(data)
                pub.publish(data)
            rate.sleep()
            frame_count += 1

    finally:
        pipeline.stop()
        pub.publish("Shutting down.")

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass