import cv2 as cv
import numpy as np
import os



focal_length_pixels = 359  # Example focal length in pixels
baseline_meters = 0.15  # Example baseline in meters

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_stereo_calibration(filename):
    # Load stereo calibration parameters from an XML file
    cv_file = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    cameraMatrix1 = cv_file.getNode("cameraMatrix1").mat()
    distCoeffs1 = cv_file.getNode("distCoeffs1").mat()
    cameraMatrix2 = cv_file.getNode("cameraMatrix2").mat()
    distCoeffs2 = cv_file.getNode("distCoeffs2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    cv_file.release()
    focal_length_left = cameraMatrix1[0, 0]
    focal_length_right = cameraMatrix2[0, 0]
    print("Focal length of left camera:", focal_length_left)
    print("Focal length of right camera:", focal_length_right)
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T

ensure_dir('images/Right')
ensure_dir('images/Left')

# Load stereo calibration data
calib_filename = 'stereo_calibration_data.xml'  # Update this path as needed
cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T = load_stereo_calibration(calib_filename)

# Open the cameras
cap_left = cv.VideoCapture(1)  # Adjust camera index as needed
cap_right = cv.VideoCapture(2)  # Adjust camera index as needed



cap_left.set(cv.CAP_PROP_AUTOFOCUS, 0)
cap_right.set(cv.CAP_PROP_AUTOFOCUS, 0)

cap_left.set(cv.CAP_PROP_FOCUS, 50)
cap_right.set(cv.CAP_PROP_FOCUS, 50)

# Wait for cameras to initialize
ret_left, frame_left = cap_left.read()
ret_right, frame_right = cap_right.read()

if not ret_left or not ret_right:
    print("Failed to capture images from both cameras.")
    cap_left.release()
    cap_right.release()
    exit()

# Compute rectification transforms for both cameras
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    cameraMatrix1, distCoeffs1,
    cameraMatrix2, distCoeffs2,
    frame_left.shape[:2], R, T, alpha=0)

map1_left, map2_left = cv.initUndistortRectifyMap(
    cameraMatrix1, distCoeffs1, R1, P1,
    frame_left.shape[:2], cv.CV_16SC2)
map1_right, map2_right = cv.initUndistortRectifyMap(
    cameraMatrix2, distCoeffs2, R2, P2,
    frame_right.shape[:2], cv.CV_16SC2)

# StereoSGBM Matcher for disparity calculation
stereo = cv.StereoSGBM_create(
    numDisparities=16, blockSize=15
    # minDisparity=0,
    # numDisparities=16*3,
    # blockSize=5,
    # P1=8 * 3 * 5**2,
    # P2=32 * 3 * 5**2,
    # disp12MaxDiff=1,
    # uniquenessRatio=15,
    # speckleWindowSize=0,
    # speckleRange=2,
    # preFilterCap=63,
    # mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

while True:
    # Capture frames from both cameras
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("Failed to capture images from both cameras.")
        break

    # Convert to grayscale for disparity calculation
    gray_left = cv.cvtColor(frame_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(frame_right, cv.COLOR_BGR2GRAY)

    # # Rectify images
    rectified_left = cv.remap(gray_left, map1_left, map2_left, cv.INTER_LINEAR)
    rectified_right = cv.remap(gray_right, map1_right, map2_right, cv.INTER_LINEAR)


    # Compute the disparity map
    disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0

    ##
    #
    #
    #
    #
    depth_map = np.zeros_like(disparity)
    non_zero_disparity = disparity > 0  # Avoid division by zero
    depth_map[non_zero_disparity] = (focal_length_pixels * baseline_meters) / disparity[non_zero_disparity]
    #
    #
    #
    #
    #
    #
    ## 
    
    
    norm_disparity = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)


    roi_size = 100  # Size of the square's side
    height, width = depth_map.shape
    center_x, center_y = width // 2, height // 2
    roi = depth_map[center_y - roi_size // 2:center_y + roi_size // 2,
                    center_x - roi_size // 2:center_x + roi_size // 2]

    # Calculate the average depth in the ROI
    average_depth = np.mean(roi[roi > 0])  # Ignore zero values
    print(f"Average depth in the ROI: {average_depth:.2f} meters")


    # Apply a color map to the normalized disparity map
    color_disparity = cv.applyColorMap(norm_disparity, cv.COLORMAP_JET)
    cv.imshow('Color Disparity', color_disparity)

    # Display the rectified images and disparity map
    cv.imshow('Rectified Left Camera POV', rectified_left)
    cv.imshow('Rectified Right Camera POV', rectified_right)
    print(depth_map)
    cv.imshow('Disparity Map', (depth_map*250/depth_map.max()).astype(int))

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit program when 'q' key is pressed
        break

# Release video capture objects and close windows
cap_left.release()
cap_right.release()
cv.destroyAllWindows()
