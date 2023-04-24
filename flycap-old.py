import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')



import numpy as np
import cv2
from mss import mss
from PIL import Image
import cv2.aruco as aruco
import os
import pickle
import math

import argparse
from cameraHandler import run_camera_loop



import socket
from threading import Thread
import time
import json
from socketUtil import *

HOST = "0.0.0.0" # bind to all interfaces
PORT = 6666

PLANNER_HOST = "127.0.0.1"
PLANNER_PORT = 6667

send_grid = False
grid = None
plan = None




# Constant parameters used in Aruco methods
# ARUCO_PARAMETERS = aruco.DetectorParameters()
# ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
# detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)


ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)



CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 11 

# Create grid board object we're using in our stream
# CHARUCO_BOARD = aruco.CharucoBoard_create(
#         squaresX=CHARUCOBOARD_COLCOUNT,
#         squaresY=CHARUCOBOARD_ROWCOUNT,
#         squareLength=0.04,
#         markerLength=0.02,
#         dictionary=ARUCO_DICT)
# CHARUCO_BOARD = aruco.CharucoBoard(
CHARUCO_BOARD = aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        # size=(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
        squareLength=18.83/1000,#0.04,
        markerLength=(18.83 * 16/20) / 1000,#0.02,
        dictionary=ARUCO_DICT)

markerLength = 146.05#50.8#(18.83 * 16/20) / 1000#0.05;
objPoints = np.array([
    [-markerLength/2, markerLength/2, 0],
    [markerLength/2, markerLength/2, 0],
    [markerLength/2, -markerLength/2, 0],
    [-markerLength/2, -markerLength/2, 0]
])


def markerCenter(corners):
    corners = corners.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

    return np.array([cX, cY])


def drawArucoMarkers(img, corners, ids):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            # draw the ArUco marker ID on the image
            cv2.putText(img, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
    return img


def drawProjected(img, objectPoints, rvec, tvec, cameraMatrix, distCoeffs, color, thickness):
    projected, _ = cv2.projectPoints(objectPoints = np.array(objectPoints), rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    points = np2list(projected)

    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        cv2.line(img, np2cvi(p1), np2cvi(p2), color, thickness)

    return img


def flatten(arr):
    return np.array(arr).flatten()

def np2cvi(p):
    return (int(p[0]), int(p[1]))

def np2list(projected_to_cam):
    return [p[0] for p in projected_to_cam]



corner1 = None
corner2 = None
corner1_vals = []
corner2_vals = []
target_id = 70
obstacle_ids = [66, 42, 69, 1, 2, 3, 4, 5, 6, 7]
tracking = {}


def vector_avg(vectors, ind):
    total = np.array([0., 0., 0.])
    for v in vectors:
        total += flatten(v[ind])
    return total / len(vectors)


def calculate_error(corner1, corner2, img):
    c1_center = flatten(corner1[2])
    c2_center = flatten(corner2[2])
    span = c2_center - c1_center

    rotationMatrix, _ = cv2.Rodrigues(corner1[1])

    x_in_plane = np.array([1, 0, 0])
    y_in_plane = np.array([0, 1, 0])
    
    inv_rotation = np.linalg.inv(rotationMatrix)
    
    diagonal_in_plane = inv_rotation.dot(c2_center - c1_center)
    sizeX = diagonal_in_plane.dot(x_in_plane)
    sizeY = diagonal_in_plane.dot(y_in_plane)
    # print(sizeX,sizeY)
    # print(diagonal_in_plane)
    rvec = corner1[1]
    tvec = corner1[2]
    
    projected, _ = cv2.projectPoints(objectPoints = np.array([[sizeX, sizeY, 0]]), rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
    calculated_c2 = np.array(projected[0][0])
    # (topLeft, topRight, bottomRight, bottomLeft) = corners
    real_c2 = markerCenter(corner2[0])
    if img is not None:
        cv2.line(img, np2cvi(calculated_c2), np2cvi(real_c2), (0, 0, 255), 4)


        p1, _ = cv2.projectPoints(objectPoints = np.array([[0., 0., 0]]), rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        p1 = np.array(p1[0][0])
        p2, _ = cv2.projectPoints(objectPoints = np.array([[diagonal_in_plane[0], diagonal_in_plane[1], 0]]), rvec=rvec, tvec=tvec, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
        p2 = np.array(p2[0][0])
        cv2.line(img, np2cvi(p1), np2cvi(p2), (0, 255, 0), 4)
        
    error = np.linalg.norm(real_c2 - calculated_c2)
    return error

def get_current_corners():
    corner1 = None
    corner2 = None
    error = None

    if len(corner1_vals) > 0:
        # corner1 = (corner1_vals[-1][0], vector_avg(corner1_vals, 1), vector_avg(corner1_vals, 2))
        corner1 = corner1_vals[-1]
    if len(corner2_vals) > 0:
        # corner2 = (corner2_vals[-1][0], vector_avg(corner2_vals, 1), vector_avg(corner2_vals, 2))
        corner2 = corner2_vals[-1]

    if corner1 is not None and corner2 is not None:
        error = calculate_error(corner1=corner1, corner2=corner2, img=None)
    return (corner1, corner2, error)

def coordsInPlane(centerInWorld, originInWorld, inv_rotation):
    x_in_plane = np.array([1, 0, 0])
    y_in_plane = np.array([0, 1, 0])

    targ_center = flatten(centerInWorld)
    # print(targ_center)
    target_displacement = targ_center - originInWorld
    target_in_plane = inv_rotation.dot(target_displacement)
    targX = target_in_plane.dot(x_in_plane)
    targY = target_in_plane.dot(y_in_plane)

    return np.array([targX, targY])

b = None
def process_frame(img):
    global corner1
    global corner2
    global b
    global tracking
    # global corner1_vals
    # global corner2_vals

    # img = cv2.undistort(img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    # corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)

    # Refine detected markers
    # Eliminates markers not part of our board, adds missing markers to the board
    # corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
    corners, ids, rejectedImgPoints, recoveredIds = aruco.refineDetectedMarkers(
            image = gray,
            board = CHARUCO_BOARD,
            detectedCorners = corners,
            detectedIds = ids,
            rejectedCorners = rejectedImgPoints,
            cameraMatrix = cameraMatrix,
            distCoeffs = distCoeffs)  

    target = None

    if corners is not None and len(corners) > 0:

        # Outline all of the markers detected in our image
        img = drawArucoMarkers(img, corners, ids)
        img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))

        corner1 = None
        corner2 = None

        for (corner, id) in zip(corners, ids):
            id = id[0]
            retval, rvec, tvec = cv2.solvePnP(
                objectPoints=objPoints, 
                imagePoints=corner, 
                cameraMatrix=cameraMatrix, 
                distCoeffs=distCoeffs)           
            img = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, markerLength/2)#0.015)

            data = (corner, rvec, tvec)
            if id == 24:
                corner1 = data
                # corner1_vals.append(data)
            elif id == 87:
                corner2 = data
                # corner2_vals.append(data)
            elif id == target_id:
                target = data
            
            if id in obstacle_ids or id == target_id:
                curr_corner1, curr_corner2, curr_error = get_current_corners()
                if curr_corner1 is not None:
                    error = calculate_error(corner1=curr_corner1, corner2=data, img=None)
                    error_old = 0
                    if id in tracking:
                        old = (corner, tracking[id][1], tracking[id][2])
                        error_old = calculate_error(corner1=curr_corner1, corner2=old, img=None)
                        # print(id, ":", error, "vs", error_old)
                    if error <= 40 and (id not in tracking or error <= error_old):
                        tracking[id] = data

        # if (corner1 is not None and (corner2 is not None or len(corner2_vals) > 0)) or corner2 is not None and (corner1 is not None or len(corner1_vals) > 0):
        if corner1 is not None and corner2 is not None:
            curr_corner1, curr_corner2, curr_error = get_current_corners()

            error = calculate_error(corner1=corner1, corner2=corner2, img=None)
            # error = calculate_error(corner1=corner1 if corner1 is not None else corner1_vals[-1], corner2=corner2 if corner2 is not None else corner2_vals[-1], img=img)
            # print("ATTEMPT ERROR: ", error)
            
            if error <= 20 and (curr_corner1 is None or curr_corner2 is None or error <= curr_error): # good, and don't accept higher error
                # print("added",error)
                corner1_vals.append(corner1)
                corner2_vals.append(corner2)
            else:
                corner1 = None
                corner2 = None

    corner1, corner2, error = get_current_corners()

    if corner1 is not None and corner2 is not None:
        # print("CURRENT ERROR: ", error)

        rvec = corner1[1]
        tvec = corner1[2]
        
        c1_center = flatten(corner1[2])
        c2_center = flatten(corner2[2])
        span = c2_center - c1_center

        # print("found", span, np.linalg.norm(span))
        up = np.array([0, 0, markerLength/2])

        rotationMatrix, _ = cv2.Rodrigues(rvec)

        # inPlane = np.array([np.linalg.norm(horizontal), 0, 0])
        # rotated = rotationMatrix.dot(inPlane)
        x_in_plane = np.array([1, 0, 0])
        y_in_plane = np.array([0, 1, 0])
        
        inv_rotation = np.linalg.inv(rotationMatrix)
        
        
        diagonal_in_plane = inv_rotation.dot(c2_center - c1_center)
        sizeX = diagonal_in_plane.dot(x_in_plane)
        sizeY = diagonal_in_plane.dot(y_in_plane)
        gridSize = int(math.floor(abs(sizeX) / 330))#7

        # print("Diagonal: ", int(np.linalg.norm(span)), "mm")
        # print("Size X: ", abs(int(sizeX)), "mm")
        # print("Size Y: ", abs(int(sizeY)), "mm")

        stepX = sizeX / gridSize
        stepY = (-1 if sizeY < 0 else 1) * abs(stepX)#sizeY / gridSize
        gridSizeY = math.ceil(abs(sizeY / stepY))#gridSize

        fullSizeY = gridSizeY * stepY
        

        gridColor = (255, 255, 0)
        gridThickness = 3

        for i in range(-1, gridSize+1):
            gridY = -1
            gridX = i
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)

            gridY = gridSizeY
            gridX = i
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)
        for i in range(-1, gridSizeY+1):
            gridX = -1
            gridY = i
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)

            gridX = gridSize
            gridY = i
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)
        
        for i in range(max(gridSize, gridSizeY)+1):
            if i <= gridSize:
                x = i * stepX
                # draw y line
                drawProjected(img, [
                    [x, 0, 0],
                    # [x, 0, markerLength/2],
                    # [x, 0, 0],
                    [x, fullSizeY, 0],
                ], rvec, tvec, cameraMatrix, distCoeffs, gridColor, gridThickness)

            if i <= gridSizeY:
                y = i * stepY
                # draw x line
                drawProjected(img, [
                    [0, y, 0],
                    [sizeX, y, 0],
                ], rvec, tvec, cameraMatrix, distCoeffs, gridColor, gridThickness)

        

        
        # if target is not None:
            # targ_center = flatten(target[2])
            # # print(targ_center)
            # target_displacement = targ_center - c1_center
            # target_in_plane = inv_rotation.dot(target_displacement)
            # targX = target_in_plane.dot(x_in_plane)
            # targY = target_in_plane.dot(y_in_plane)

        for obs_id in obstacle_ids:
            if obs_id in tracking:
                obstacle = tracking[obs_id]
                obs_center = coordsInPlane(obstacle[2], c1_center, inv_rotation)
                # error = calculate_error(corner1, obstacle, img=img)
                # print("ERROR OBS", error)
                gridX = int(obs_center[0] / stepX)
                gridY = int(obs_center[1] / stepY)
                
                drawProjected(img, [
                    [gridX * stepX, gridY * stepY, 0],
                    [(gridX+1) * stepX, gridY * stepY, 0],
                    [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                    [(gridX) * stepX, (gridY+1) * stepY, 0],
                    [gridX * stepX, gridY * stepY, 0]
                ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)

        targetForwardInPlane = None
        targetAngle = None
        if target is not None:
            target_rvec = target[1]
            targ_center = coordsInPlane(target[2], c1_center, inv_rotation)
            targetRotationMatrix, _ = cv2.Rodrigues(target_rvec)
            forward = np.array([0, 1, 0])
            targetForwardInWorld = targetRotationMatrix.dot(forward)
            targetForwardInPlane = inv_rotation.dot(targetForwardInWorld)

            targetAngle = math.atan2(targetForwardInPlane[0], targetForwardInPlane[1]) * 180 / math.pi
            print("ANGLE", targetAngle)
            # print(targetForwardInPlane)
            # drawProjected(img, [
            #     [targ_center[0], targ_center[1], 0],
            #     [targ_center[0] + targetForwardInPlane[0]*markerLength, targ_center[1] + targetForwardInPlane[1]*markerLength, 0],
            # ], rvec, tvec, cameraMatrix, distCoeffs, (0, 255, 255), 5)

        if target_id in tracking:
            target = tracking[target_id]
            targ_center = coordsInPlane(target[2], c1_center, inv_rotation)
            targX = targ_center[0]
            targY = targ_center[1]
            
            gridX = int(targX / stepX)
            gridY = int(targY / stepY)

            
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 255, 0), gridThickness)
            
            drawProjected(img, [
                [0, 0, 0],
                [0, 0, markerLength/2],
                [targX, targY, markerLength/2],
                [targX, targY, 0],
                [targX, targY, markerLength/2],
                [sizeX, sizeY, markerLength/2],
                [sizeX, sizeY, 0],
                [targX, targY, 0],
                [0, 0, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 255, 255), 2)

            target_error = calculate_error(corner1, target, img=img)

            if targetForwardInPlane is not None and targetAngle is not None:
                drawProjected(img, [
                    [targ_center[0], targ_center[1], 0],
                    [targ_center[0] + targetForwardInPlane[0]*markerLength*2, targ_center[1] + targetForwardInPlane[1]*markerLength*2, 0],
                ], rvec, tvec, cameraMatrix, distCoeffs, (255, 0, 255), 3)
            # if b is None or target_error <= b:
            #     b = target_error
            # print("ERROR TARGET: ", target_error)
            # print("best", b)



    # Only try to find CharucoBoard if we found markers
    if ids is not None and len(ids) > 10:

        # Get charuco corners and ids from detected aruco markers
        response, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=CHARUCO_BOARD)

        # Require more than 20 squares
        if response is not None and response > 20:
            # Estimate the posture of the charuco board, which is a construction of 3D space based on the 2D video 

            pose, rvec, tvec = aruco.estimatePoseCharucoBoard(
                    charucoCorners=charuco_corners, 
                    charucoIds=charuco_ids, 
                    board=CHARUCO_BOARD, 
                    cameraMatrix=cameraMatrix, 
                    distCoeffs=distCoeffs, 
                    rvec=np.empty(1),
                    tvec=np.empty(1))
            if pose:
                # Draw the camera posture calculated from the gridboard
                #img = aruco.drawAxis(img, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
                img = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
        
    # Display our image
    cv2.imshow('QueryImage', img)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False


calib_file = './calibration.pckl'

# Check for camera calibration data
if not os.path.exists(calib_file):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open(calib_file, 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove " + calib_file + " and recalibrate your camera with calibrateCamera.py.")
        exit()



# ap = argparse.ArgumentParser()
# ap.add_argument("-c", "--camera", type=int, required=False, default=0,
# 	help="camera type (default 0)")
# args = vars(ap.parse_args())

# camera_type = args["camera"]


run_camera_loop(process_frame)