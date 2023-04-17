# The following code is used to watch a video stream, detect Aruco markers, and use
# a set of markers to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import numpy
import cv2
import cv2.aruco as aruco


# Constant parameters used in Aruco methods
ARUCO_PARAMETERS = aruco.DetectorParameters()
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMETERS)

# Create grid board object we're using in our stream
# board = aruco.GridBoard_create(
#         markersX=2,
#         markersY=2,
#         markerLength=0.09,
#         markerSeparation=0.01,
#         dictionary=ARUCO_DICT)

# # Create vectors we'll be using for rotations and translations for postures
# rvecs, tvecs = None, None

#cam = cv2.VideoCapture('gridboardiphonetest.mp4')
cam = cv2.VideoCapture(0)

while(cam.isOpened()):
    # Capturing each frame of our video stream
    ret, img = cam.read()
    if ret == True:
        # grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Detect Aruco markers
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        
        # # Make sure all 5 markers were detected before printing them out
        # # if ids is not None and len(ids) == 5:
        # # Print corners and ids to the console
        # for i, corner in zip(ids, corners):
        #     print('ID: {}; Corners: {}'.format(i, corner))

        # # Outline all of the markers detected in our image
        # img = aruco.drawDetectedMarkers(img, corners, borderColor=(0, 0, 255))

        #     # # Wait on this frame
        #     # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     #     break

        # # Display our image
        # cv2.imshow('QueryImage', img)

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
                print("[INFO] ArUco marker ID: {}".format(markerID))

    # show the output image
    cv2.imshow("Image", img)
    # cv2.waitKey(0)

    # Exit at the end of the video on the 'q' keypress
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
