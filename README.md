# Posenet ROS Wrapper
## Tasks:
* Display keypoints and skeleton on a blank canvas in Recorder.py draw_pose()
* Implement self label and auto label functions (can be the same)
* add arguments for:
  * variable time - how long do you want to record all frames for
  * single shot mode - just take a single frame at a given time delay instead of all frames. Also label it.
### issues:
* tuple error with keypoint_scores, keypoint_coords being restricted to a 1 width array. Directly clashes with draw_skel_and_kp() in utils.py.
* issue reading loaded ndarray and writing label to it.
