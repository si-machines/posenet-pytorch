# Posenet ROS Wrapper
## Tasks:
* talker.py - preprocessing
  * only send frame if all 17 points are found
* recorder.py - data sampling
  * take more data sampling and add to set
* classifier.py - preprocessing
  * get the average coordinate of points (the center of gravity) and subtract it against each coordinate to get a "centered" image (*utils.py*)
    * apply to both Recorder.py images and Classifier.py images
  * look into different types of clustering algorithms after knn
* utils.py - function that displays all the different pose types found in frame_data_example.

### issues:
* how to normalize difference between taller and shorter people?
* how to normalize difference in position globally?
* how to normalize difference in perspective?
