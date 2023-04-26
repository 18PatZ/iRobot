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

from statistics import median

import argparse
from cameraHandler import run_camera_loop



import socket
from threading import Thread
import time
import json
from socketUtil import *

HOST = "0.0.0.0" # bind to all interfaces
PORT = 6666

PLANNER_HOST = "10.224.123.203"
PLANNER_PORT = 6667

send_grid = False
sent_grid = False
grid = None
policies = None
schedule = None




# Constant parameters used in Aruco methods
# ARUCO_PARAMETERS = aruco.DetectorParameters()
# ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
# detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)


ARUCO_PARAMETERS = aruco.DetectorParameters_create()
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
ARUCO_DICT_4 = aruco.Dictionary_get(aruco.DICT_4X4_50)



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

# 129 and 154
markerLength = 146.05#50.8#(18.83 * 16/20) / 1000#0.05;
markerLength4p = 129 # side by side 4x4 markers
markerLength4 = 154 # single 4x4 marker



def calcObjPoints(sideLength):
    return np.array([
        [-sideLength/2, sideLength/2, 0],
        [sideLength/2, sideLength/2, 0],
        [sideLength/2, -sideLength/2, 0],
        [-sideLength/2, -sideLength/2, 0]
    ])


objPoints = calcObjPoints(markerLength)


def markerCenter(corners):
    corners = corners.reshape((4, 2))
    (topLeft, topRight, bottomRight, bottomLeft) = corners

    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
    cY = int((topLeft[1] + bottomRight[1]) / 2.0)

    return np.array([cX, cY])

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

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
target_angles = []
# obstacle_ids = [66, 42, 69, 1, 2, 3, 4, 5, 6, 7]


id4_top = 20
id4_bottom = 21
id4 = 30

obstacle_ids = [1, 66, 5, 6, 7, 42, 69]
# obstacle_ids = []#[42, 1, 2, 3, 4, 5]
tracking = {}
gridPos = {}

# obs_pos = [(2,1),(5,1),(8,1),(8,0),(2,4),(5,4),(8,4)]
obs_pos = [(3,1),(6,1),(8,1),(8,0),(3,4),(6,4),(8,4)]

gSizeX = 0
gSizeY = 0

goalCoords = None

winner = False


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


def transformGridCoords(pos, sizeX, sizeY, mult = 1):
    (x, y) = pos
    # x, y = int(y), int(x)
    x, y = int(mult*x), int(mult*y)
    # sX = int(sizeY)
    # sY = int(sizeX)
    sX = int(mult*sizeX)
    sY = int(mult*sizeY)
    
    if sX < 0:
        x = (-sX-1) - x
        sX = -sX

    if sY < 0:
        y = -y
        sY = -sY
    else:
        y = (sY-1) - y

    # x, y = int(mult * x), int(mult * y)

    return (x, y, sX, sY)
        
def forwardVecToAngle(forward):
    return math.atan2(forward[0], forward[1]) * 180 / math.pi

def extractTargetAngle(id, target, c1_center, inv_rotation):
    target_rvec = target[1]
    targ_center = coordsInPlane(target[2], c1_center, inv_rotation)
    targetRotationMatrix, _ = cv2.Rodrigues(target_rvec)
    forward = np.array([0, 1, 0])
    up = np.array([0, 0, 1])
    
    targetForwardInWorld = targetRotationMatrix.dot(forward)
    targetForwardInPlane = inv_rotation.dot(targetForwardInWorld)

    targetUpInWorld = targetRotationMatrix.dot(up)
    targetUpInPlane = inv_rotation.dot(targetUpInWorld)
    if targetUpInPlane[2] <= 0:
        # print("Bad target",id,"Z axis!", targetUpInPlane[2])
        return (None, None)
    else:
        targetAngle = forwardVecToAngle(targetForwardInPlane)

    return (targetAngle, targetForwardInPlane)



def process_frame(img):
    global corner1
    global corner2
    global gridPos
    global tracking
    global grid
    global send_grid
    global gSizeX
    global gSizeY
    global goalCoords
    global winner
    # global corner1_vals
    # global corner2_vals

    # img = cv2.undistort(img, cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect Aruco markers
    # corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    
    corners4, ids4, _ = aruco.detectMarkers(gray, ARUCO_DICT_4, parameters=ARUCO_PARAMETERS)
    if corners4 is not None:
        # print(corners.shape)
        # print(corners4.shape)
        corners.extend(corners4)
        # corners = np.append(corners, corners4, axis=0)
    if ids4 is not None and len(ids4) > 0:
        # print(ids4)
        ids = np.append(ids, ids4, axis=0)
        # ids.extend(ids4)
    # corners.extend(corners4)
    # ids.extend(ids4)

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
    target_top = None
    target_bottom = None

    if corners is not None and len(corners) > 0:

        # Outline all of the markers detected in our image
        img = drawArucoMarkers(img, corners, ids)
        img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))

        corner1 = None
        corner2 = None

        for (corner, id) in zip(corners, ids):
            id = id[0]

            mLength = markerLength
            if id == id4_bottom or id == id4_top:
                mLength = markerLength4p
            elif id == id4:
                mLength = markerLength4

            objPoints = calcObjPoints(mLength)
            
            retval, rvec, tvec = cv2.solvePnP(
                objectPoints=objPoints, 
                imagePoints=corner, 
                cameraMatrix=cameraMatrix, 
                distCoeffs=distCoeffs)           
            img = cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rvec, tvec, mLength/2)#0.015)

            data = (corner, rvec, tvec)
            if id == 24:
                corner1 = data
                # corner1_vals.append(data)
            elif id == 87:
                corner2 = data
                # corner2_vals.append(data)
            elif id == target_id or id == id4:
                target = data
            elif id == id4_top:
                target_top = data
            elif id == id4_bottom:
                target_bottom = data
            
            if id in obstacle_ids or id == target_id or id == id4_top or id == id4_bottom or id == id4:
                curr_corner1, curr_corner2, curr_error = get_current_corners()
                if curr_corner1 is not None:
                    error = calculate_error(corner1=curr_corner1, corner2=data, img=None)
                    error_old = 0
                    if id in tracking:
                        old = (corner, tracking[id][1], tracking[id][2])
                        error_old = calculate_error(corner1=curr_corner1, corner2=old, img=None)
                        # print(id, ":", error, "vs", error_old)
                    # if id == id4:
                    #     print("error",error)
                    if error <= 40 and (id not in tracking or error <= error_old):
                        tracking[id] = data

        # if (corner1 is not None and (corner2 is not None or len(corner2_vals) > 0)) or corner2 is not None and (corner1 is not None or len(corner1_vals) > 0):
        if corner1 is not None and corner2 is not None:
            curr_corner1, curr_corner2, curr_error = get_current_corners()

            error = calculate_error(corner1=corner1, corner2=corner2, img=None)
            # error = calculate_error(corner1=corner1 if corner1 is not None else corner1_vals[-1], corner2=corner2 if corner2 is not None else corner2_vals[-1], img=img)
            # print("ATTEMPT ERROR: ", error)
            
            if error <= 20 and (curr_corner1 is None or curr_corner2 is None or error <= curr_error) and not sent_grid: # good, and don't accept higher error
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
        gridCellSize = 250#280#330
        gridSize = int(math.floor(abs(sizeX) / gridCellSize))#7

        

        # print("Diagonal: ", int(np.linalg.norm(span)), "mm")
        # print("Size X: ", abs(int(sizeX)), "mm")
        # print("Size Y: ", abs(int(sizeY)), "mm")

        stepX = sizeX / gridSize
        stepY = (-1 if sizeY < 0 else 1) * abs(stepX)#sizeY / gridSize
        gridSizeY = math.ceil(abs(sizeY / stepY))#gridSize

        fullSizeY = gridSizeY * stepY

        gSizeX = np.sign(sizeX) * gridSize
        gSizeY = np.sign(fullSizeY) * gridSizeY
        

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

                # drawProjected(img, [
                #     [gridX * stepX, gridY * stepY, 0],
                #     [(gridX+1) * stepX, gridY * stepY, 0],
                #     [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                #     [(gridX) * stepX, (gridY+1) * stepY, 0],
                #     [gridX * stepX, gridY * stepY, 0]
                # ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)

                gridPos[obs_id] = (gridX, gridY)

        for obs in obs_pos:
            gridX = (abs(int(gSizeX))-1) - obs[0]
            gridY = (abs(int(gSizeY))-1) - obs[1]
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), gridThickness)

        if goalCoords is not None:
            gridX = goalCoords[0]
            gridY = goalCoords[1]
            drawProjected(img, [
                [gridX * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, gridY * stepY, 0],
                [(gridX+1) * stepX, (gridY+1) * stepY, 0],
                [(gridX) * stepX, (gridY+1) * stepY, 0],
                [gridX * stepX, gridY * stepY, 0]
            ], rvec, tvec, cameraMatrix, distCoeffs, (255, 0, 255), gridThickness)

        targetForwardInPlane = None
        targetAngle = None
        if target is not None:
            targetAngle, targetForwardInPlane = extractTargetAngle(target_id, target, c1_center, inv_rotation)
        
        if target_top is not None or target_bottom is not None:
            targetForwardInPlane = np.array([0., 0., 0.])
            n = 0

            if target_top is not None:
                targetAngle, targetForwardInPlaneT = extractTargetAngle(id4_top, target_top, c1_center, inv_rotation)
                targetForwardInPlane += targetForwardInPlaneT
                n += 1
            if target_bottom is not None:
                targetAngle, targetForwardInPlaneB = extractTargetAngle(id4_bottom, target_bottom, c1_center, inv_rotation)
                targetForwardInPlane += targetForwardInPlaneB
                n += 1

            targetForwardInPlane = normalize(targetForwardInPlane/n)
            targetAngle = forwardVecToAngle(targetForwardInPlane)

            if id4_top in tracking and id4_bottom in tracking:
                top_center = coordsInPlane(tracking[id4_top][2], c1_center, inv_rotation)
                bottom_center = coordsInPlane(tracking[id4_bottom][2], c1_center, inv_rotation)
                right_vec = normalize(top_center - bottom_center)
                print(forwardVecToAngle(right_vec))
                up_vec = np.array([0, 0, 1])
                forward_vec = np.cross(up_vec, right_vec)
                targetForwardInPlane = normalize((targetForwardInPlane + forward_vec) / 2)

                targetAngleNew = forwardVecToAngle(targetForwardInPlane)
                print("headings",int(targetAngle),"vs",int(targetAngleNew))
                targetAngle = targetAngleNew

        if targetAngle is not None:
            target_angles.append(targetAngle)
            # print("ANGLE", targetAngle)
            # print(targetForwardInPlane)
            # drawProjected(img, [
            #     [targ_center[0], targ_center[1], 0],
            #     [targ_center[0] + targetForwardInPlane[0]*markerLength, targ_center[1] + targetForwardInPlane[1]*markerLength, 0],
            # ], rvec, tvec, cameraMatrix, distCoeffs, (0, 255, 255), 5)

        if target_id in tracking or id4_top in tracking or id4_bottom in tracking or id4 in tracking:
            if id4_top in tracking and id4_bottom in tracking:
                targetT = tracking[id4_top]
                targetB = tracking[id4_bottom]
                targ_centerT = coordsInPlane(targetT[2], c1_center, inv_rotation)
                targ_centerB = coordsInPlane(targetB[2], c1_center, inv_rotation)
                targ_center = (targ_centerT + targ_centerB) / 2
            else:
                if target_id in tracking:
                    target = tracking[target_id]
                elif id4 in tracking:
                    target = tracking[id4]
                elif id4_top in tracking:
                    target = tracking[id4_top]
                elif id4_bottom in tracking:
                    target = tracking[id4_bottom]
                targ_center = coordsInPlane(target[2], c1_center, inv_rotation)

            targX = targ_center[0]
            targY = targ_center[1]
            
            gridX = int(targX / stepX)
            gridY = int(targY / stepY)

            gridPos[target_id] = (targX / stepX, targY / stepY)


            if gSizeX is not None and goalCoords is not None:
                # tX, tY, sX, sY = transformGridCoords(gridPos[target_id], gSizeX, gSizeY)
                # print("Win check:",(gridX, gridY),"vs",goalCoords)
                if gridX == goalCoords[0] and gridY == goalCoords[1]:
                    winner = True
                    # print("\n\n\n\nWINNER!!!!!!!\n\n\n\n")

            
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

            # target_error = calculate_error(corner1, target, img=img)

            if targetForwardInPlane is not None and targetAngle is not None:
                drawProjected(img, [
                    [targ_center[0], targ_center[1], 0],
                    [targ_center[0] + targetForwardInPlane[0]*markerLength*2, targ_center[1] + targetForwardInPlane[1]*markerLength*2, 0],
                ], rvec, tvec, cameraMatrix, distCoeffs, (0, 0, 255), 2)

                angle = median(target_angles)
                y = math.cos(angle * math.pi / 180)
                x = math.sin(angle * math.pi / 180)
                forward = np.array([x, y, 0])

                drawProjected(img, [
                    [targ_center[0], targ_center[1], 0],
                    [targ_center[0] + forward[0]*markerLength*2, targ_center[1] + forward[1]*markerLength*2, 0],
                ], rvec, tvec, cameraMatrix, distCoeffs, (255, 0, 255), 3)
            # if b is None or target_error <= b:
            #     b = target_error
            # print("ERROR TARGET: ", target_error)
            # print("best", b)

        if not sent_grid:
            grid_ready = True
            # for obs_id in obstacle_ids:
            #     if obs_id not in tracking:
            #         grid_ready = False
            if (target_id not in tracking and id4 not in tracking and (id4_top not in tracking or id4_bottom not in tracking)) or targetAngle is None:
                grid_ready = False

            if grid_ready:
                # print("GRID READY")
                _, _, sX, sY = transformGridCoords((0, 0), gSizeX, gSizeY)
                new_grid = [[0 for x in range(sX)] for y in range(sY)]

                # tX, tY, _, _ = transformGridCoords(gridPos[target_id], sizeX, sizeY)
                # grid[tY][tX] = 3

                # for obs_id in obstacle_ids:
                for obs in obs_pos:
                    # oX = (int(abs(gSizeX))-1) - obs[0]
                    # oY = (int(abs(gSizeY))-1) - obs[1]
                    oX = obs[0]
                    oY = obs[1]
                    # print(gSizeX, gSizeY, obs, oX, oY)  
                    # oX, oY, _, _ = transformGridCoords(gridPos[obs_id], gSizeX, gSizeY)
                    new_grid[oY][oX] = 1 # obstacle

                goalPos = (int(abs(gSizeX)-1), int(abs(gSizeY / 2)))
                gX, gY, _, _ = transformGridCoords(goalPos, gSizeX, gSizeY)
                new_grid[gY][gX] = 2 # goal

                # goalCoords = (gX, gY)
                goalCoords = goalPos

                grid = new_grid
                send_grid = True



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






def plannerInterface():
    global send_grid
    global grid
    global policies
    global schedule
    global sent_grid

    planner = socket.socket()
    print("Connecting to planner at " + PLANNER_HOST + ":" + str(PLANNER_PORT) + "...")
    planner.connect((PLANNER_HOST, PLANNER_PORT))
    print("Connected!")

    while not send_grid:
        time.sleep(1)

    send_grid = False
    sent_grid = True

    grid_json = json.dumps(grid)

    print("Sending " + str(len(grid)) + "x" + str(len(grid[0])) + " grid to planner...")
    
    planner.send("PLAN".encode())
    sendMessage(planner, grid_json)

    print("Sent! Waiting for plan...")
    plan_json = receiveMessage(planner)
    plan = json.loads(plan_json)

    #to_send = {"Policy": policyToJsonFriendly(policy), "Schedule": schedule}
    policies = jsonFriendlyToPolicy(plan["Policy"])
    schedule = plan["Schedule"]
    print("Received policies:", policies)
    print("Received schedule:", schedule)


time_per_step = 2
checkin_index = 0
def sendCheckin(sock):
    global checkin_index
    global target_angles

    while True:
        if len(target_angles) > 0 and target_id in gridPos and schedule is not None:
            angle = median(target_angles)
            tX, tY, sX, sY = transformGridCoords(gridPos[target_id], gSizeX, gSizeY, mult=1)

            if tX >= 0 and tX < sX and tY >= 0 and tY < sY:

                if winner:
                    print("Sending win notification!")
                    sendMessage(sock, "exit")
                    return True

                angle += 360
                
                message = str(int(checkin_index)) + " " + str(int(tX)) + " " + str(int(tY)) + " " + str(int(angle))
                print("CHECKIN",message)

                ind = checkin_index
                if ind >= len(schedule)-1:
                    ind = len(schedule)-1

                state = (int(tX), int(tY))
                action = policies[ind][state] if state in policies[ind] else "in wall, up to agent"
                print("  Agent should be doing:", action)

                sendMessage(sock, message)

                received = sock.recv(3).decode()
                if received == "ACK":
                    print("ACK received, continuing")
                
                next_stride = int(schedule[ind])
                to_wait = time_per_step * next_stride
                print("  Next observation in", to_wait, "seconds...")

                for i in range(next_stride):
                    target_angles.clear()
                    time.sleep(time_per_step)
                # time.sleep(to_wait)

                checkin_index += 1

                return False

        time.sleep(0.05)
    
    

    
def serverInterface():


    # while True:
    #     sendCheckin(None)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s.bind((HOST, PORT))
    print("Server bound to " + HOST + ":" + str(PORT))
    s.listen()
    print("Listening for inbound connections...")
    conn, addr = s.accept()

    #with conn:
    print("Agent connection established from " + str(addr))

    target_angles.clear()

    while True:
        if sendCheckin(conn):
            conn.close()
            s.close()
            break
        # data = conn.recv(1024)
        # # if not data:
        # #     break
        # received = data.decode()
        # print("Received: " + received)

        # to_send = ""
        # for i in range(5000):
        #     to_send += received
        # # conn.send(("ping " + received).encode())
        # print("sending back", len(to_send),"bytes")
        # conn.send(len(to_send).to_bytes(2, 'little', signed=False))
        # conn.send(to_send.encode())

        # if received == "exit":
        #     conn.close()
        #     s.close()
        #     break




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

planner_thread = Thread(target=plannerInterface, args=[])
planner_thread.start()

server_thread = Thread(target=serverInterface, args=[])
server_thread.start()

run_camera_loop(process_frame)

planner_thread.join()
server_thread.join()
