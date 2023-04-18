# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import cv2
import time
import os

import argparse
from cameraHandler import run_camera_loop


path = "./output"
if not os.path.exists(path):
    os.makedirs(path)


prev_frame_time = time.time()
frame = 0

interval = 0.5 # per second
interval_frames = int(interval * 30) # 30 fps


def process_frame(img):
    global frame
    global prev_frame_time

    frame += 1
    if frame % interval_frames == 0:
        num = int(frame / interval_frames)
        cv2.imwrite(path + "/" + "img_" + str(num) + ".jpg", img)
        cv2.putText(img, "SAVED " + str(num), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 100, 0), 2, cv2.LINE_AA)

    
    dt = time.time() - prev_frame_time
    fps = 1 / dt
    prev_frame_time = time.time()

    cv2.putText(img, "FPS " + str(int(fps)), (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (100, 255, 0), 2, cv2.LINE_AA)
        
    # Display our image
    cv2.imshow('Video Feed', img)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


run_camera_loop(process_frame)
