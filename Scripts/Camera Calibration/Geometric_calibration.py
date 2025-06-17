<<<<<<< HEAD
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
image_folder = './Group_1'
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
=======
import cv2
import numpy as np
import glob
import os

# === PARAMETERS ===
CHECKERBOARD = (8,5)  # Number of inner corners (columns, rows)
square_size = 25       # Square size in mm

print("📐 Checkerboard size:", CHECKERBOARD)
print("🧱 Square size:", square_size, "mm")

# Termination criteria for refining corner locations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points grid like: (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by actual square size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# === Load Calibration Images ===
image_folder = './'
images = glob.glob(os.path.join(image_folder, '*.JPG'))  # Adjust extension if needed

print(f"📷 Found {len(images)} images in '{image_folder}'")

if not images:
    raise FileNotFoundError(f"No images found in folder: {image_folder}")

os.makedirs('output_corners', exist_ok=True)  # Save annotated images

for i, fname in enumerate(images, 1):
    print(f"\n🔍 Processing image {i}/{len(images)}: {fname}")
    img = cv2.imread(fname)
    if img is None:
        print(f"❌ Failed to read image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print("✅ Checkerboard detected.")
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and save the corners image
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        out_path = os.path.join('output_corners', os.path.basename(fname))
        cv2.imwrite(out_path, img)
        print(f"💾 Saved corner image to: {out_path}")
    else:
        print("⚠️ Checkerboard not found in this image.")

# === Perform Camera Calibration ===
print("\n📸 Starting camera calibration...")

if len(objpoints) < 1:
    raise RuntimeError("❌ Not enough valid calibration images with detected checkerboard corners.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("✅ Calibration complete!")

# === Evaluate Calibration ===
print("\n📏 Calculating mean reprojection error...")
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    print(f"  - Image {i+1}: Error = {error:.4f}")

mean_error /= len(objpoints)

# === Output Calibration Results ===
print("\n=== 📤 Calibration Results ===")
print("🎯 Camera Matrix:\n", camera_matrix)
print("\n🎯 Distortion Coefficients:\n", dist_coeffs.ravel())
print("\n📉 Mean Reprojection Error: {:.4f}".format(mean_error))



# === OPTIONAL: Convert focal length from pixels to millimeters ===
sensor_width_mm = 35.9     # e.g., Nikon D850 
sensor_height_mm = 23.9
image_width_px = gray.shape[1]  # width from image
image_height_px = gray.shape[0]  # height from image

fx_pixels = camera_matrix[0, 0]
fy_pixels = camera_matrix[1, 1]

fx_mm = fx_pixels * (sensor_width_mm / image_width_px)
fy_mm = fy_pixels * (sensor_height_mm / image_height_px)

print("\n=== 📐 Focal Length in Millimeters ===")
print(f"📏 fx = {fx_mm:.2f} mm")
print(f"📏 fy = {fy_mm:.2f} mm")


# === Save Results to File ===
np.savez('calibration_results.npz',
         camera_matrix=camera_matrix,
         dist_coeffs=dist_coeffs,
         rvecs=rvecs,
         tvecs=tvecs)
print("\n💾 Calibration results saved to 'calibration_results.npz'")
>>>>>>> 07904be9fb8ec70be1f7482ca483b7af1d6e5c33
