# Posenet ROS Wrapper
## Tasks:
* recorder.py - data sampling
  - [x] take more data sampling and add to set
  - [x] call center_of_gravity function - either center keypoints immediately after receiving data points (drawn pose will be centered) or only center before saving into frame_data_example (currently, Classifier.py centers keypoints after grabbing original data from frame_data_example)
  - [ ] have option for user to choose to list saved data
* classifier.py - preprocessing
  - [ ] look into different types of clustering algorithms after knn
* Explore UT demo
  - [x] get algorithm to run on Poli2
    - [x] display name of pose on external monitor(?)
    - [ ] display name of pose on Poli2's screen
  - [ ] change robot face in response to specific poses
* optimizing KNN algorithm
  - [ ] asymmetric tolerance - flip every image horizontally and run KNN twice; take the best result
  - [ ] limb scaling - take an average ratio of limbs given current data; for new data take the length of one limb (say the torso) and scale all limbs according to that
* Data
  - [ ] reconfigure talker.py to accept still frames of poses and send them to Recorder.py to build a better data set


### issues:
* how to normalize difference between taller and shorter people?
* how to normalize difference in position globally?
* how to normalize difference in perspective?
* melodic setup:
  * `ImportError: No module named converter.tfjs2pytorch` coming from posenet.load_model(101) - rosrun is running the executable from a weird place
  * similar issue with Recorder.py and trying to save data to *frame_data_example*
