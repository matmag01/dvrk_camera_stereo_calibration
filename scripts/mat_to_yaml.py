import os
import sys
import scipy.io
import numpy as np
import cv2
from typing import List, Tuple
import yaml

dynamic_path = os.path.abspath(__file__+"/../")
sys.path.append(dynamic_path)

def _get_dist_model(d, cam_model="plumb_bob"):
    return cam_model

# Save calib parameter in yaml file (ROS)
def write_yaml(name, d, k, r, p, size, cam_model="plumb_bob"):
    def format_mat(x, precision):
        return ("[%s]" % (
            np.array2string(x, precision=precision, suppress_small=True, separator=", ")
              .replace("[", "").replace("]", "").replace("\n", "\n        ")
        ))

    dist_model = _get_dist_model(d, cam_model)

    assert k.shape == (3, 3)
    assert r.shape == (3, 3)
    assert p.shape == (3, 4)
    calmessage = "\n".join([
        "image_width: %d" % size[0],
        "image_height: %d" % size[1],
        "camera_name: " + name,
        "camera_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: " + format_mat(k, 5),
        "distortion_model: " + dist_model,
        "distortion_coefficients:",
        "  rows: 1",
        "  cols: %d" % d.size,
        "  data: [%s]" % ", ".join("%8f" % x for x in d.flat),
        "rectification_matrix:",
        "  rows: 3",
        "  cols: 3",
        "  data: " + format_mat(r, 8),
        "projection_matrix:",
        "  rows: 3",
        "  cols: 4",
        "  data: " + format_mat(p, 5),
        ""
    ])

    yaml_name = f"{name}.yaml"
    with open(yaml_name, "w") as f:
        f.write(calmessage)
    print(f"File saved: {yaml_name}")


def mat_to_yaml(mat_path, img_shape: Tuple[int,int]):
    """
    Convert Matlab stereo calibration .mat to ROS YAML files.
    img_shape: (width, height) of images
    """
    mat = scipy.io.loadmat(mat_path)
    # --- Extract matrices from .mat ---
    K1 = np.array(mat['intrinsicMatrix1'])  # MATLAB intrinsicMatrix is transposed vs OpenCV
    K2 = np.array(mat['intrinsicMatrix2'])
    D1 = np.array(mat['distortionCoefficients1'])
    D2 = np.array(mat['distortionCoefficients2'])
    R = np.array(mat['rotationOfCamera2'])
    T = np.array(mat['translationOfCamera2']).reshape(3,1)

    K1 = np.array(K1, dtype=np.float64)
    K2 = np.array(K2, dtype=np.float64)
    D1 = np.array(D1, dtype=np.float64)
    D2 = np.array(D2, dtype=np.float64)
    R = np.array(R, dtype=np.float64)
    T = np.array(T, dtype=np.float64)
    # Stereo rectification to get P1 and P2
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=img_shape,
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )

    # Save YAML files
    write_yaml("left_m", D1, K1, R1, P1, img_shape)
    write_yaml("right_m", D2, K2, R2, P2, img_shape)
    print("YAML files generated!")

if __name__=="__main__":
    mat_path = os.path.join(dynamic_path, "calib_result.mat")
    img_shape = (1300, 1024)  # width, height -> cambia in base alle tue immagini
    mat_to_yaml(mat_path, img_shape)
