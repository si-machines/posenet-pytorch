# Posenet ROS Wrapper
## Tasks:
* talker.py - preprocessing
  * only send frame if all 17 points are found
* recorder.py - data sampling
  * take more data sampling and add to set
  * call center_of_gravity function - either center keypoints immediately after receiving data points (drawn pose will be centered) or only center before saving into frame_data_example (currently, Classifier.py centers keypoints after grabbing original data from frame_data_example) 
* classifier.py - preprocessing
  * look into different types of clustering algorithms after knn

### issues:
* how to normalize difference between taller and shorter people?
* how to normalize difference in position globally?
* how to normalize difference in perspective?
* melodic setup:
  * `ImportError: No module named converter.tfjs2pytorch` coming from posenet.load_model(101) - rosrun is running the executable from a weird place
  * similar issue with Recorder.py and trying to save data to *frame_data_example*


