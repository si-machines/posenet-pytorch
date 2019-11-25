# Posenet ROS Wrapper
## Tasks:
* Display keypoints and skeleton on a blank canvas in Recorder.py draw_pose()
* Implement self label and auto label functions (can be the same)
### issues:
* tuple error with keypoint_scores, keypoint_coords being restricted to a 1 width array. Directly clashes with draw_skel_and_kp() in utils.py.
* issue reading loaded ndarray and writing label to it.
### Notes:
* To read pickled data in the frame_data_example folder, start up python3 in the console.

```
import numpy as np
array = np.load("PATH_TO_FILE", allow_pickle=True)

# print entire data structure
print(array)

# print frame id
print(array[0])

# print pose_scores
print(array[1])

# print keypoint_scores
print(array[2])

# print keypoint_coords
print(array[3])
```
