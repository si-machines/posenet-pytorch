import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import torch
import cv2
import time
import argparse
import numpy as np

import posenet

# import libraries for realsense D435
import pyrealsense2 as rs

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--realsense', type=int, default=0)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()

def reshape(lst, n):
    return [lst[i*n:(i+1)*n] for i in range(len(lst)//n)]

def main():
    model = posenet.load_model(args.model)
    # model = model.cuda()
    output_stride = model.output_stride

    if args.realsense is 1:
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
                    color_image, realsense=True, scale_factor=args.scale_factor, output_stride=output_stride)

                with torch.no_grad():
                    input_image = torch.Tensor(input_image)
                    # input_image = torch.Tensor(input_image).cuda()

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
                data = ["Pose Score", pose_scores[0],
                        "Keypoint Coords", keypoint_coords[0],
                        "Keypoint Scores", keypoint_scores[0]]
                print(data)
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

    else:
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image)
                # input_image = torch.Tensor(input_image).cuda()

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

            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))

if __name__ == "__main__":
    main()
