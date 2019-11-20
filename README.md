# Posenet ROS Wrapper
## Tasks:
* Display keypoints and skeleton on a blank canvas in Recorder.py draw_pose()
### issues:
* tuple error with keypoint_scores, keypoint_coords being restricted to a 1 width array. Directly clashes with draw_skel_and_kp() in utils.py.
