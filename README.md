# dVRK Stereo Camera Calibration

This repository provides a stereo camera calibration pipeline tailored for the [da Vinci Research Kit (dVRK)](https://dvrk.readthedocs.io/main/).  
It allows you to calibrate stereo endoscope cameras, save calibration results in **ROS YAML** format, and export intrinsic/extrinsic parameters to **JSON**.

## Usage

- Capture images of a checkerboard pattern with both left and right cameras with  ```save_sync_images.py```
- Run calibration with  ```stereo_calib.py```. Good results if projection error is below 1.
- Output files:
  - ```left.yaml``` and ```right.yaml```: calibration files in ROS format
  - ```stereo_calib_params.json```: JSON file with camera matrices, distortion, R, T, E, F
 - To test, run ```stereo_calib_test.py```. This script draws equally spaced horizontal lines on the image to check epipolar alignment.

## Credits
This project is based on [original repository](https://github.com/jackzhy96/stereo_camera_calibration/tree/6c9a81ffd16867440bad7af3152f95f1053db2b4)
- Copyright (c) 2025 Haoying (Jack) Zhou
- Licensed under the MIT License
  
Modifications by Matteo Magnani (2025).

