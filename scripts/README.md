## PoseNet Pytorch

This repository contains a PyTorch implementation (multi-pose only) of the Google TensorFlow.js Posenet model.

This port is based on my Tensorflow Python (https://github.com/rwightman/posenet-python) conversion of the same model. An additional step of the algorithm was performed on the GPU in this implementation so it is faster and consumes less CPU (but more GPU). On a GTX 1080 Ti (or better) it can run over 130fps.

Further optimization is possible as the MobileNet base models have a throughput of 200-300 fps.

### Usage

There are three demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

### Credits

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet

This port and my work is in no way related to Google.

The Python conversion code that started me on my way was adapted from the CoreML port at https://github.com/infocom-tpo/PoseNet-CoreML

### TODO (someday, maybe)
* More stringent verification of correctness against the original implementation
* Performance improvements (especially edge loops in 'decode.py')
* OpenGL rendering/drawing
* Comment interfaces, tensor dimensions, etc
* Implement batch inference for image_demo
* Create a training routine and add models with more advanced CNN backbones
