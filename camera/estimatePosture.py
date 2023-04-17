# The following code is used to watch a video stream, detect a Charuco board, and use
# it to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy as np
import cv2
import cv2.aruco as aruco
import os
import pickle



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


# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
        exit()

# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 11 

# Create grid board object we're using in our stream
# CHARUCO_BOARD = aruco.CharucoBoard_create(
#         squaresX=CHARUCOBOARD_COLCOUNT,
#         squaresY=CHARUCOBOARD_ROWCOUNT,
#         squareLength=0.04,
#         markerLength=0.02,
#         dictionary=ARUCO_DICT)
CHARUCO_BOARD = aruco.CharucoBoard(
        #squaresX=CHARUCOBOARD_COLCOUNT,
        #squaresY=CHARUCOBOARD_ROWCOUNT,
        size=(CHARUCOBOARD_COLCOUNT, CHARUCOBOARD_ROWCOUNT),
        squareLength=18.83/1000,#0.04,
        markerLength=(18.83 * 16/20) / 1000,#0.02,
        dictionary=ARUCO_DICT)

# Create vectors we'll be using for rotations and translations for postures
rvecs, tvecs = None, None


markerLength = (18.83 * 16/20) / 1000#0.05;
objPoints = np.array([
    [-markerLength/2, markerLength/2, 0],
    [markerLength/2, markerLength/2, 0],
    [markerLength/2, -markerLength/2, 0],
    [-markerLength/2, -markerLength/2, 0]
])




# cam = cv2.VideoCapture('charucoboard-test-at-kyles-desk.mp4')
cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, QueryImg = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(QueryImg, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
        # Refine detected markers
        # Eliminates markers not part of our board, adds missing markers to the board
        corners, ids, rejectedImgPoints, recoveredIds = detector.refineDetectedMarkers(
                image = gray,
                board = CHARUCO_BOARD,
                detectedCorners = corners,
                detectedIds = ids,
                rejectedCorners = rejectedImgPoints,
                cameraMatrix = cameraMatrix,
                distCoeffs = distCoeffs)   

        # Outline all of the markers detected in our image
        QueryImg = drawArucoMarkers(QueryImg, corners, ids)
        QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

        #single_rvecs, single_tvecs = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)           

        for corner in corners:
            retval, rvec, tvec = cv2.solvePnP(
                objectPoints=objPoints, 
                imagePoints=corner, 
                cameraMatrix=cameraMatrix, 
                distCoeffs=distCoeffs)           
            QueryImg = cv2.drawFrameAxes(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.015)

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
                    #QueryImg = aruco.drawAxis(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
                    QueryImg = cv2.drawFrameAxes(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
            
        # Display our image
        cv2.imshow('QueryImage', QueryImg)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
