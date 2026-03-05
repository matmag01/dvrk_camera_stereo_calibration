#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from glob import glob
import json
from typing import List, Tuple
from cv2 import aruco

repo_path = os.path.abspath(__file__+"/../../")
# print(dynamic_path)
sys.path.append(repo_path)
dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)
import yaml

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
        glob(os.path.join(data_folder, "frame1_*.png")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    right_img_path = sorted(
        glob(os.path.join(data_folder, "frame2_*.png")),
        key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0])
    )
    return left_img_path, right_img_path

def _get_dist_model(d, cam_model="plumb_bob"):
    return cam_model

def resize_matrix(matrix, orig_size, new_size):
    """
    Resize the camera intrinsic matrix when the image size changes.

    Args:
        matrix: The original camera intrinsic matrix (3x3).
        orig_size: Tuple of original image size (width, height).
        new_size: Tuple of new image size (width, height).

    Returns:
        The resized camera intrinsic matrix (3x3).
    """
    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]
    resized_matrix = matrix.copy()
    resized_matrix[0, 0] *= scale_x  # fx
    resized_matrix[1, 1] *= scale_y  # fy
    resized_matrix[0, 2] *= scale_x  # cx
    resized_matrix[1, 2] *= scale_y  # cy
    return resized_matrix

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


if __name__=='__main__':
    
    # ARUCO BOARD CREATION
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)

    board = aruco.CharucoBoard(
        (5, 7),                # squaresX, squaresY
        0.01,                  # square length (m)
        0.006,                 # marker length (m)
        dictionary
    )
    
    
    data_folder = os.path.join(repo_path, "scripts","Images_GStream_charuco")
    print(f"Data folder: {data_folder}")
    left_img_path, right_img_path = load_img(data_folder)
    
    detector_params = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    all_objpoints = []
    all_imgpointsL = []
    all_imgpointsR = []
    count = 0
    
    for imgL_path, imgR_path in zip(left_img_path, right_img_path):

        imgL_raw = cv2.imread(imgL_path)
        imgR_raw = cv2.imread(imgR_path)
        
        if imgL_raw  is None or imgR_raw  is None:
            print(f"Skipping pair: {imgL_path}, {imgR_path} --> img not found")
            continue
        
        grayL, grayR = cv2.cvtColor(imgL_raw, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR_raw, cv2.COLOR_BGR2GRAY)

        # --- Detect ArUco ---
        cornersL, idsL, _ = aruco_detector.detectMarkers(grayL)
        cornersR, idsR, _ = aruco_detector.detectMarkers(grayR)  
        debugL = imgL_raw.copy()
        debugR = imgR_raw.copy() 
        
        if idsL is None or idsR is None:
            print(f"Skipping pair: {imgL_path}, {imgR_path} --> no ArUco markers detected")
            cv2.imshow("Left Debug", debugL)
            cv2.imshow("Right Debug", debugR)
            cv2.waitKey(200)
            continue  
        
        if idsL is not None:
            aruco.drawDetectedMarkers(debugL, cornersL, idsL)

        if idsR is not None:
            aruco.drawDetectedMarkers(debugR, cornersR, idsR)  

        # --- Interpolate Charuco corners ---
        retL, charucoCornersL, charucoIdsL = aruco.interpolateCornersCharuco(
            cornersL, idsL, grayL, board
        )

        retR, charucoCornersR, charucoIdsR = aruco.interpolateCornersCharuco(
            cornersR, idsR, grayR, board
        )
        
        if retL < 4 or retR < 4:
            cv2.imshow("Left Debug", debugL)
            cv2.imshow("Right Debug", debugR)
            cv2.waitKey(200)
            continue
        
        # Disegna corner charuco
        aruco.drawDetectedCornersCharuco(debugL, charucoCornersL, charucoIdsL)
        aruco.drawDetectedCornersCharuco(debugR, charucoCornersR, charucoIdsR)
    
        common_ids = np.intersect1d(charucoIdsL.flatten(), charucoIdsR.flatten())
        if len(common_ids) < 4:
            cv2.imshow("Left Debug", debugL)
            cv2.imshow("Right Debug", debugR)
            cv2.waitKey(200)
            continue
        
        count += 1
        imgPtsL, imgPtsR, objPts = [], [], []
    
        for cid in common_ids:
            idxL = np.where(charucoIdsL == cid)[0][0]
            idxR = np.where(charucoIdsR == cid)[0][0]

            imgPtsL.append(charucoCornersL[idxL])
            imgPtsR.append(charucoCornersR[idxR])
            corners3D = board.getChessboardCorners()
            objPts.append(corners3D[cid])
            
            
            # Disegna ID comune in rosso
            ptL = tuple(charucoCornersL[idxL][0].astype(int))
            ptR = tuple(charucoCornersR[idxR][0].astype(int))

            cv2.circle(debugL, ptL, 5, (0,0,255), -1)
            cv2.circle(debugR, ptR, 5, (0,0,255), -1)
            
            print(f"Frame valido - punti usati: {len(common_ids)}")

        
        all_objpoints.append(np.array(objPts, dtype=np.float32))
        all_imgpointsL.append(np.array(imgPtsL, dtype=np.float32))
        all_imgpointsR.append(np.array(imgPtsR, dtype=np.float32))
        
        cv2.imshow("Left Debug", debugL)
        cv2.imshow("Right Debug", debugR)

        key = cv2.waitKey(300)
        if key == 27:  # ESC
            break

    # Check:
    if not all_objpoints:
        raise ValueError("No valid checkerboard corners found!")
    
    img_shape = cv2.imread(left_img_path[0], cv2.IMREAD_GRAYSCALE).shape[::-1]

    # Left camera calibration
    _, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(all_objpoints, all_imgpointsL, img_shape, None, None)

    # Right camera calibration
    _, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(all_objpoints, all_imgpointsR, img_shape, None, None)

    SINGLE_CAM_REPROJ_ERR_THRESH = 1.5
    print("Frame used: ", count)
    # Evaluate per-image reprojection errors for single cameras
    errsL = []
    errsR = []
    
    for i, (rvec, tvec) in enumerate(zip(rvecsL, tvecsL)):
        proj, _ = cv2.projectPoints(all_objpoints[i], rvec, tvec, mtxL, distL)
        err = cv2.norm(all_imgpointsL[i], proj, cv2.NORM_L2) / len(proj)
        errsL.append(err)
    for i, (rvec, tvec) in enumerate(zip(rvecsR, tvecsR)):
        proj, _ = cv2.projectPoints(all_objpoints[i], rvec, tvec, mtxR, distR)
        err = cv2.norm(all_imgpointsR[i], proj, cv2.NORM_L2) / len(proj)
        errsR.append(err)
        
    # Reject the outliners
    keep_indices = []
    for i in range(len(all_objpoints)):
        if errsL[i] < SINGLE_CAM_REPROJ_ERR_THRESH and errsR[i] < SINGLE_CAM_REPROJ_ERR_THRESH:
            keep_indices.append(i)

    if len(keep_indices) < len(all_objpoints):
        print(f"\n Rejecting {len(all_objpoints) - len(keep_indices)} outlier frames based on reprojection error.")

    filtered_objpoints = [all_objpoints[i] for i in keep_indices]
    filtered_imgpointsL = [all_imgpointsL[i] for i in keep_indices]
    filtered_imgpointsR = [all_imgpointsR[i] for i in keep_indices]

    # Recalibrate single cameras with filtered images
    retL2, mtxL2, distL2, rvecsL2, tvecsL2 = cv2.calibrateCamera(
        filtered_objpoints, filtered_imgpointsL, img_shape, None, None,
    )

    retR2, mtxR2, distR2, rvecsR2, tvecsR2 = cv2.calibrateCamera(
        filtered_objpoints, filtered_imgpointsR, img_shape, None, None,
    )

    print("\nLeft single-cam RMS after outlier removal:", retL2)
    print("Right single-cam RMS after outlier removal:", retR2)
    print(f"Frames used for stereo calibration: {len(filtered_objpoints)} / {len(all_objpoints)}")
    
    # stereo calibration
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)
    flags = cv2.CALIB_FIX_INTRINSIC
    retStereo, camMatL, distCoefL, camMatR, distCoefR, R, T, E, F = cv2.stereoCalibrate(
        filtered_objpoints,
        filtered_imgpointsL,
        filtered_imgpointsR,
        mtxL2,
        distL2,
        mtxR2,
        distR2,
        img_shape,
        criteria=criteria_stereo,
        flags=flags
    )

    print(f"Stereo Calibration RMS error: {retStereo:.4f}")
    baseline = np.linalg.norm(T)
    print(f"Baseline: {baseline:.4f} m")

    # Stereo Rectification
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camMatL, distCoefL, camMatR, distCoefR, img_shape, R, T, alpha=0
    )

    # # Resize to 1080x720
    # new_size = (1080, 720)
    # camMatLresize = resize_matrix(camMatL, img_shape, new_size)
    # camMatRresize = resize_matrix(camMatR, img_shape, new_size)
    #     # Stereo Rectification
    # R1Resize, R2Resize, P1Resize, P2Resize, QResize, roi1Resize, roi2Resize = cv2.stereoRectify(
    #     camMatLresize, distCoefL, camMatRresize, distCoefR, new_size, R, T, alpha=0
    # )

    # YAML saving for ROS
    write_yaml("left", distCoefL, camMatL, R1, P1, img_shape)
    write_yaml("right", distCoefR, camMatR, R2, P2, img_shape)
    # write_yaml("left_resized", distCoefL, camMatLresize, R1Resize, P1Resize, new_size)
    # write_yaml("right_resized", distCoefR, camMatRresize, R2Resize, P2Resize, new_size)
    print("YAML files saved for ROS.")

    # save to json
    calib_data = {
        "cameraMatrixL": camMatL.tolist(),
        "distCoeffsL": distCoefL.tolist(),
        "cameraMatrixR": camMatR.tolist(),
        "distCoeffsR": distCoefR.tolist(),
        "R": R.tolist(),
        "T": T.tolist(),
        "E": E.tolist(),
        "F": F.tolist(),
        "rms_left": float(retL2),
        "rms_right": float(retR2),
        "rms_stereo": float(retStereo),
        "baseline_m": float(baseline)
    }

    with open("stereo_calib_params_new.json", "w") as f:
        json.dump(calib_data, f, indent=4)

    print("Stereo calibration parameters saved to 'stereo_calib_params_new.json'")