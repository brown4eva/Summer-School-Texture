import cv2
import numpy as np
import glob
import os

# === PARAMETERS ===
CHECKERBOARD = (9, 6)  # number of internal corners in the checkerboard (columns, rows)
square_size = 25  # in mm or any consistent unit

# Termination criteria for corner subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3d point in real world
imgpoints = []  # 2d points in image plane

# === Load Calibration Images ===
image_folder = './calibration_images'
images = glob.glob(os.path.join(image_folder, '*.jpg'))  # Change to .png if needed

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # If found, refine corners and add to list
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# === Perform Camera Calibration ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# === Evaluate Calibration ===
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

# === Output Calibration Results ===
print("Camera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs.ravel())
print("\nMean reprojection error:", mean_error)

# === Optional: Save to file ===
np.savez('calibration_results.npz', 
         camera_matrix=camera_matrix, 
         dist_coeffs=dist_coeffs,
         rvecs=rvecs, 
         tvecs=tvecs)
