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
    K1 = np.array(mat['intrinsicMatrix1']).T  # MATLAB intrinsicMatrix is transposed vs OpenCV
    K2 = np.array(mat['intrinsicMatrix2']).T
    D1 = np.array(mat['distortionCoefficients1']).flatten()
    D2 = np.array(mat['distortionCoefficients2']).flatten()
    R = np.array(mat['rotationOfCamera2'])
    T = np.array(mat['translationOfCamera2']).flatten()

    # Stereo rectification to get P1 and P2

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_shape, R, T, alpha=0
    )

    # Save YAML files
    write_yaml("left", D1, K1, R1, P1, img_shape)
    write_yaml("right", D2, K2, R2, P2, img_shape)
    print("YAML files generated!")

if __name__=="__main__":
    mat_path = os.path.join(dynamic_path, "calib_result.mat")
    img_shape = (1300, 1024)  # width, height -> cambia in base alle tue immagini
    mat_to_yaml(mat_path, img_shape)
