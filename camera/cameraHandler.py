import numpy as np
import cv2
from mss import mss
from PIL import Image

import argparse


def run_camera_loop(process_frame):

    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--camera", type=int, required=False, default=0,
        help="camera type (default 0)")
    args = vars(ap.parse_args())

    camera_type = args["camera"]


    if camera_type == 0: # normal video camera
        cam = cv2.VideoCapture(0)

        while(cam.isOpened()):
            # Capturing each frame of our video stream
            ret, img = cam.read()
            
            if ret == True:
                if process_frame(img):
                    cv2.destroyAllWindows()
                    break

        return



    if camera_type == 1 or camera_type == 2: # flycap screen capture
        if camera_type == 1:
            corner1 = np.array([613, 85])
            corner2 = np.array([2212, 1363])
        else:
            # corner1 = np.array([485, 55+35])
            # corner2 = np.array([1440, 820+35])

            corner1 = np.array([-250+95, 50+28])
            corner2 = np.array([-250+95+958, 800])
        dim = corner2 - corner1

        monitorOffset = 2560#0

        bounding_box = {'top': int(corner1[1]), 'left': int(corner1[0]+monitorOffset), 'width': int(dim[0]), 'height': int(dim[1])}

        sct = mss()

        while True:
            sct_img = sct.grab(bounding_box)
            img = np.array(sct_img)

            if img.shape[2] == 4: # BGRA, remove alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if process_frame(img):
                cv2.destroyAllWindows()
                break

        return



    if camera_type == 3: # valve index exterior
        cam = cv2.VideoCapture(0)

        while(cam.isOpened()):
            # Capturing each frame of our video stream
            ret, img = cam.read()

            img = img[:, int(img.shape[1]/2):img.shape[1]] # crop in half, since its two eyes side by side
            
            if ret == True:
                if process_frame(img):
                    cv2.destroyAllWindows()
                    break

        return


