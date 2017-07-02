# Precise Vehicle Tracking Using Panoramic Camera

This repository contains the script and a sample data for multi-vehicle tracking in a panoramic scene. The data was collected on July 23rd, 2016 in Tucsan, AZ. The test footage lasted for about 7 minutes and 22 seconds and recorded detailed movement of 20 vehicles. The script is used to extract and visualize the trajectories of those 20 vehicles in the scene.

## Installation

The shell script is tested in Linux environment has the following dependencies:
+ ffmpeg
+ Python 2.7
+ OpenCV Python binding
+ Scipy stack
+ scikit-learn

## Usage

To run the video processing pipline,
```
>> mkdir az_0726_expB
>> wget https://www.dropbox.com/s/urcmdzqwdnbdkkg/original_video.mp4?dl=0
>> ./run.sh
```

## Full Experiments
This repository only contains data for one single experiment. However, a total of 9 experiments were conducted in Tuscan in July 23rd. To access the full stack of video footages, please go to [here](https://uofi.box.com/s/0xphjvaiekl8wldrwkypmb6yfzflwass).

## Dataset
The vehicle movement data generated using the script, combined with the simultaneous fuel rate measured from a OBD scanner, is publicly available [here](https://uofi.box.com/v/trafficwavedata).


## Contact
+ Author: Fangyu Wu, Coordinated Science Laboratory, UIUC
+ Email: fwu10(at)illinois.edu
+ Web: fangyuwu.com
