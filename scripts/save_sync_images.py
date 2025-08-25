#!/usr/bin/env python3
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospy
from scipy.spatial.transform import Rotation as Rot
import copy
import time
import json
import os
import sys

dynamic_path = os.path.abspath(__file__+"/../")
# print(dynamic_path)
sys.path.append(dynamic_path)

img_folder = os.path.join(dynamic_path, 'image')

if not os.path.exists(img_folder):
    # Create folder if not exisist
    os.makedirs(img_folder)

bridge = CvBridge()

class dvrk_img_save:
    def __init__(self):
        rospy.init_node('save_images', anonymous=True)
        self.camera_left_topic = '/.../left/image_raw' # --> Add name of stereo_rig_name instead of ...
        self.camera_right_topic = '/.../right/image_raw' # --> Add name of stereo_rig_name instead of ...
        self.sub_topic_camera_left = rospy.Subscriber(self.camera_left_topic, Image, self.camera_left_sub, queue_size=1)
        self.sub_topic_camera_right = rospy.Subscriber(self.camera_right_topic, Image, self.camera_right_sub, queue_size=1)
        self.camera_left_img = None
        self.camera_right_img = None
        self.count = 0
    
    # Subscriber of left camera
    def camera_left_sub(self, msg):
        try:
            self.camera_left_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(f"Error retrieving data from camera left: {e}")

    # Subscriber right camera
    def camera_right_sub(self, msg):
        try:
            self.camera_right_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(f"Error retrieving data from camera right: {e}")

    # Save the images
    def save_img(self, folder_path, img_left, img_right):
        # Check images:
        if img_left is None or img_right is None:
            rospy.logwarn("Images not received yet, skipping save.")
            return
        
        img_left = copy.deepcopy(img_left)
        img_right = copy.deepcopy(img_right)

        # png savage --> better quality of images
        img_left_name = f'{self.count:03d}_left.png'
        img_right_name = f'{self.count:03d}_right.png'
        img_left_path = os.path.join(folder_path, img_left_name)
        img_right_path = os.path.join(folder_path, img_right_name)

        # Savage of images
        cv2.imwrite(img_left_path, img_left)
        cv2.imwrite(img_right_path, img_right)

    def get_img(self):
        return self.camera_left_img, self.camera_right_img

    def run_application(self, feq):
        rate = rospy.Rate(feq)
        time.sleep(1)
        while not rospy.is_shutdown():
            img_left, img_right = self.get_img()
            print('saving')
            self.save_img(img_folder, img_left, img_right)
            self.count += 1
            input("Press Enter to Continue")
            # 60 images to calibrate
            if self.count >= 45:
                break
            rate.sleep()

if __name__=="__main__":
    frequency = 1
    sub_cls = dvrk_img_save()
    sub_cls.run_application(frequency)
    print(f"Saved {sub_cls.count} images.")