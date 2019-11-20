import sys
import importlib
import torch
import cv2
import time
import argparse
import numpy as np

import posenet

# import libraries for realsense D435
import pyrealsense2 as rs

#parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=int, default=101)
#parser.add_argument('--realsense', type=int, default=0)
#parser.add_argument('--cam_id', type=int, default=0)
#parser.add_argument('--cam_width', type=int, default=1280)
#parser.add_argument('--cam_height', type=int, default=720)
#parser.add_argument('--scale_factor', type=float, default=0.7125)
#args = parser.parse_args()


def main():
    model = posenet.load_model(101)
    output_stride = model.output_stride

    # setup camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    start = time.time()
    frame_count = 0
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

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

            if keypoint_scores[0][0] != 0:
                print('Executing Callback at: ', time.time())
                if len(sys.argv) > 2 : # callback defined in another file at the same level
                    emit_message(pose_scores, keypoint_scores, keypoint_coords, getattr(importlib.import_module(sys.argv[2]), sys.argv[1]))
                else: # callback defined in posenet_wrapper.py
                    emit_message(pose_scores, keypoint_scores, keypoint_coords, globals()[sys.argv[1]]) # pass in all the necessary data
            
            
            # Display the image (for purposes of seeing that valid data is showing up)
            keypoint_coords *= output_scale
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)
            # Show images
            cv2.namedWindow('posenet', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:

        # Stop streaming
        pipeline.stop()
        print('Average FPS: ', frame_count / (time.time() - start))

def callback_example(pose_scores, keypoint_scores, keypoint_coords):
    print('found something...')
    #print('pose_scores: ', pose_scores)
    #print('keypoint_scores: ', keypoint_scores)
    #print('keypoint_coords: ', keypoint_coords)


def emit_message(pose_scores, keypoint_scores, keypoint_coords, callback):
    print('Running callback...')
    callback(pose_scores, keypoint_scores, keypoint_coords)


if __name__ == "__main__":
    main()
