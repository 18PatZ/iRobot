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


markerLength = 50.8#(18.83 * 16/20) / 1000#0.05;
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

        if corners is not None and len(corners) > 0:

            # Outline all of the markers detected in our image
            QueryImg = drawArucoMarkers(QueryImg, corners, ids)
            QueryImg = aruco.drawDetectedMarkers(QueryImg, corners, borderColor=(0, 0, 255))

            #single_rvecs, single_tvecs = aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)  
            corner1 = None
            corner2 = None         
            cornerA = None
            target = None

            for (corner, id) in zip(corners, ids):
                retval, rvec, tvec = cv2.solvePnP(
                    objectPoints=objPoints, 
                    imagePoints=corner, 
                    cameraMatrix=cameraMatrix, 
                    distCoeffs=distCoeffs)           
                QueryImg = cv2.drawFrameAxes(QueryImg, cameraMatrix, distCoeffs, rvec, tvec, markerLength/2)#0.015)

                data = (corner, rvec, tvec)
                if id == 24:
                    corner1 = data
                elif id == 69:
                    corner2 = data
                elif id == 70:
                    target = data
                elif id == 42:
                    cornerA = data
            
            if corner1 is not None and corner2 is not None and target is not None and cornerA is not None:
                c1_center = flatten(corner1[2])
                c2_center = flatten(corner2[2])
                targ_center = flatten(target[2])
                cA_center = flatten(cornerA[2])
                span = c2_center - c1_center

                # print("found", span, np.linalg.norm(span))
                up = np.array([0, 0, markerLength/2])

                projected_to_cam, jacobian = cv2.projectPoints(objectPoints = np.array([
                    np.array([0, 0, 0]),
                    up,
                    # up + targ_center - c1_center,
                    # up + c2_center - c1_center,
                    # c2_center - c1_center
                ]), rvec=corner1[1], tvec=corner1[2], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

                projected_to_cam2, _ = cv2.projectPoints(objectPoints = np.array([
                    up
                ]), rvec=target[1], tvec=target[2], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

                projected_to_cam3, _ = cv2.projectPoints(objectPoints = np.array([
                    up,
                    np.array([0, 0, 0])
                ]), rvec=corner2[1], tvec=corner2[2], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                
                points = np2list(projected_to_cam)
                points.extend(np2list(projected_to_cam2))
                points.extend(np2list(projected_to_cam3))

                rotationMatrix, _ = cv2.Rodrigues(corner1[1])

                horizontal = cA_center - c1_center
                inPlane = np.array([np.linalg.norm(horizontal), 0, 0])
                rotated = rotationMatrix.dot(inPlane)
                
                inv_rotation = np.linalg.inv(rotationMatrix)
                unrotated = inv_rotation.dot(horizontal)

                diagonal_in_plane = inv_rotation.dot(c2_center - c1_center)
                x_in_plane = np.array([1, 0, 0])
                y_in_plane = np.array([0, 1, 0])
                sizeX = diagonal_in_plane.dot(x_in_plane)
                sizeY = diagonal_in_plane.dot(y_in_plane)
                gridSize = 5

                print("Diagonal: ", int(np.linalg.norm(span)), "mm")
                print("Size X: ", abs(int(sizeX)), "mm")
                print("Size Y: ", abs(int(sizeY)), "mm")

                stepX = sizeX / gridSize
                stepY = sizeY / gridSize

                gridColor = (255, 255, 0)
                gridThickness = 3

                rvec = corner1[1]
                tvec = corner1[2]
                
                for i in range(gridSize+1):
                    x = i * stepX
                    # draw y line
                    drawProjected(QueryImg, [
                        [x, 0, 0],
                        [x, sizeY, 0],
                    ], rvec, tvec, cameraMatrix, distCoeffs, gridColor, gridThickness)

                    y = i * stepY
                    # draw x line
                    drawProjected(QueryImg, [
                        [0, y, 0],
                        [sizeX, y, 0],
                    ], rvec, tvec, cameraMatrix, distCoeffs, gridColor, gridThickness)


                # proj4, _ = cv2.projectPoints(objectPoints = np.array([
                #     np.array([0, 0, 0]),
                #     unrotated,
                # ]), rvec=corner1[1], tvec=corner1[2], cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)
                # cv2.line(QueryImg, np2cvi(proj4[0][0]), np2cvi(proj4[1][0]), (255, 255, 0), 2)

                # print("c1",horizontal, np.linalg.norm(horizontal))
                # print("c2",rotated, np.linalg.norm(horizontal - rotated))

                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    cv2.line(QueryImg, np2cvi(p1), np2cvi(p2), (0, 255, 255), 2)

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
