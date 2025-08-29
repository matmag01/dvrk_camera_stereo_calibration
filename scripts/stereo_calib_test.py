#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from glob import glob
import json
repo_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
sys.path.append(repo_path)
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)
from typing import List, Tuple

# def load_img(data_folder: str) -> Tuple[List[str], List[str]]:
#     left_img_path = sorted(
#         glob(os.path.join(data_folder, "left_*.png")),
#         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
#     )
#     right_img_path = sorted(
#         glob(os.path.join(data_folder, "right_*.png")),
#         key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
#     )
#     return left_img_path, right_img_path
# def load_img(data_folder: str) -> Tuple[List[str], List[str]]:
#     left_img_path = sorted(
#         glob(os.path.join(data_folder, "*_left.jpg")),
#         key=lambda x: int(os.path.basename(x).split('_')[0])
#     )
#     right_img_path = sorted(
#         glob(os.path.join(data_folder, "*_right.jpg")),
#         key=lambda x: int(os.path.basename(x).split('_')[0])
#     )
#     return left_img_path, right_img_path
def load_img(data_folder: str) -> Tuple[List[str], List[str]]:
    """
    Loads and sorts image file paths from a specified folder.

    Args:
        data_folder: The path to the directory containing the images.

    Returns:
        A tuple containing two lists of sorted image paths:
        (left_image_paths, right_image_paths).
    """
    left_img_path = sorted(
        glob(os.path.join(data_folder, "*_left.png")),
        key=lambda x: int(os.path.basename(x).split('_')[0])
    )
    right_img_path = sorted(
        glob(os.path.join(data_folder, "*_right.png")),
        key=lambda x: int(os.path.basename(x).split('_')[0])
    )
    return left_img_path, right_img_path

def load_param(file_name:str)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Load the stereo calibration parameters from a json file
    '''

    with open(file_name, "r") as f:
        data = json.load(f)
    cameraMatrixL = np.array(data["cameraMatrixL"], dtype=np.float64)
    distCoeffsL = np.array(data["distCoeffsL"], dtype=np.float64)
    cameraMatrixR = np.array(data["cameraMatrixR"], dtype=np.float64)
    distCoeffsR = np.array(data["distCoeffsR"], dtype=np.float64)
    R = np.array(data["R"], dtype=np.float64)
    T = np.array(data["T"], dtype=np.float64)
    E = np.array(data["E"], dtype=np.float64)
    F = np.array(data["F"], dtype=np.float64)

    return cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F

def draw_epipolar_lines(image, num_lines=10):
    """Draw equally spaced horizontal lines on the image to check epipolar alignment."""
    height = image.shape[0]
    step = height // num_lines
    for i in range(num_lines):
        y = i * step
        cv2.line(image, (0, y), (image.shape[1], y), (255, 0, 0), 1)
    return image

if __name__=='__main__':
    data_folder = os.path.join(repo_path, "scripts", "image")
    left_img_path, right_img_path = load_img(data_folder)
    param_path = os.path.join(dynamic_path, 'stereo_calib_params.json')
    cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, R, T, E, F = load_param(param_path)

    imgL_path = left_img_path[0]
    imgR_path = right_img_path[0]
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)
    ## test
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrixL, distCoeffsL,
        cameraMatrixR, distCoeffsR,
        imgL.shape[::-1],
        R, T,
        # alpha=0  # 0 means zoom so only valid pixels remain
    )

    # Create rectification maps for each camera
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrixL, distCoeffsL, R1, P1, imgL.shape[::-1], cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrixR, distCoeffsR, R2, P2, imgR.shape[::-1], cv2.CV_32FC1
    )

    # Example usage to remap an image:
    rect_left = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    show_left = draw_epipolar_lines(rect_left.copy(), num_lines=12)
    show_right = draw_epipolar_lines(rect_right.copy(), num_lines=12)

    cv2.imshow("left", show_left)
    cv2.imshow("right", show_right)

    cv2.waitKey(0)
    cv2.destroyAllWindows()