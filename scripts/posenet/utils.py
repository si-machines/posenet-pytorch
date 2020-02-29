import cv2
import numpy as np

import os
import posenet.constants
import math

import sys
sys.path.append("../")
import config

def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1

    if config.RESOLUTION_OVERRIDE is True:
        target_width = config.SCREEN_WIDTH
        target_height = config.SCREEN_HEIGHT
        
    return target_width, target_height


def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.transpose((2, 0, 1)).reshape(1, 3, target_height, target_width)
    return input_img, source_img, scale


def read_cap(cap, realsense=False, scale_factor=1.0, output_stride=16):
    # not using realsense D435 camera
    if realsense is False:
        res, img = cap.read()
        if not res:
            raise IOError("webcam failure")
        return _process_input(img, scale_factor, output_stride)
    # using realsense D435 camera
    else:
        return _process_input(cap, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input(img, scale_factor, output_stride)


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
    return results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img


def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):

    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    if cv_keypoints:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img

def center_of_gravity(keypoint_coords): # returns list of centered keypoints
    total_x = total_y = 0
    for x,y in zip(keypoint_coords[0::2], keypoint_coords[1::2]):
        total_x += x
        total_y += y
    center_x = total_x / float(17)
    center_y = total_y / float(17)

    adjusted_list = []
    for x,y in zip(keypoint_coords[0::2], keypoint_coords[1::2]):
        x -= center_x
        y -= center_y
        adjusted_list.append(x)
        adjusted_list.append(y)

    return tuple(adjusted_list)

def center_of_gravity_2(keypoint_coords, SCREEN_WIDTH, SCREEN_HEIGHT):
    print(keypoint_coords)
    total_x = total_y = 0
    # for x,y in zip(keypoint_coords[0::2], keypoint_coords[1::2]):
    for part in keypoint_coords:
        total_x += part[0]
        total_y += part[1]
    center_x = total_x / float(17)
    center_y = total_y / float(17)

    adjusted_list = []
    # for x,y in zip(keypoint_coords[0::2], keypoint_coords[1::2]):
    for part in keypoint_coords:
        total_x += part[0]
        total_y += part[1]
        part[0] -= center_x
        part[1] -= center_y
        entry = []
        entry.append(part[0]+SCREEN_WIDTH/2)
        entry.append(part[1]+SCREEN_HEIGHT/2)
        adjusted_list.append(entry)
    # return tuple(adjusted_list)
    print(adjusted_list)
    return adjusted_list

# example path_name = "/home/" + USER + "/catkin_ws/src/posenet_wrapper/frame_data_example"
def list_saved_poses(path_name):
    num_saved = 0
    print("Saved poses: ")
    for file_name in os.listdir(path_name):
        file_path = path_name + '/' + file_name
        try:
            frame = np.load(file_path, allow_pickle=True)
            print(frame[4])
            num_saved += 1
        except:
            pass
    print("Number of poses: ", num_saved)

# gets torso length of pose from middle top to middle bottom of torso square
def get_torso_length(coord_points):
    """
    indices of torso coordinates:
      x , y
    [10],[11] = left shoulder (5)
    [12],[13] = right shoulder (6)
    [22],[23] = left hip (11)
    [24],[25] = right hip (12)
    """
    shoulder_x = (coord_points[10] + coord_points[12]) / float(2)
    shoulder_y = (coord_points[11] + coord_points[13]) / float(2)
    hip_x = (coord_points[22] + coord_points[24]) / float(2)
    hip_y = (coord_points[23] + coord_points[25]) / float(2)
    torso_distance = math.sqrt(((shoulder_x - hip_x) ** 2) + ((shoulder_y - hip_y) ** 2))

    return torso_distance

# scale the pose to have torso of length 1
def scaling(key_points):
    """
    scales the pose according to the origin

    'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
    'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
    'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
    """
    torso_length = get_torso_length(key_points)
    ratio = float(1) / torso_length

    # multiply every xy coordinate by the ratio
    key_points = list(key_points)
    for index in range(len(key_points)):
        key_points[index] *= ratio

    key_points = tuple(key_points)

    return key_points

# flip keypoints across y-axis
def flip_vertically(key_points):
    key_points = list(key_points)
    for index in range(0,len(key_points),2):
        key_points[index] *= -1
    key_points = tuple(key_points)
    return key_points
