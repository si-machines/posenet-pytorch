# Reading files
## File names
ex. "2019-12-02_s1477_f4433"
There are 3 parts to each file name:
1. **Date** file was created
2. **Second** that the Talker.py captured this data since lifetime start
3. **Frame** that the Talker.py captured this data since lifetime start

## Opening and Reading file data
To read pickled data in the frame_data_example folder, start up python3 in the console.

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

Similarly, you can import the Recorder.py and call its *load_data(FILE_NAME)* function. This method is a little less stable.

```
import Recorder as r
recorder = r.Recorder()
recorder.load_data(FILE_NAME)
```
