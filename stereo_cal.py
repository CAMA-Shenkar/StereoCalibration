import numpy as np
import cv2 as cv
import glob

# Chessboard dimensions
chessboard_size = (7, 9)  # Adjust based on your chessboard (number of inner corners per chessboard row and column)
square_size = 20.0  # Adjust this to your chessboard square size (in millimeters or another unit)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(chessboard_size-1,2,0)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in image plane for left camera
imgpoints_right = []  # 2D points in image plane for right camera

# List of image file paths
images_left = glob.glob('images/Left/*.jpg')
images_right = glob.glob('images/Right/*.jpg')

# Ensure we have pairs of images
assert len(images_left) == len(images_right)

# Arrays to store calibration results
left_calibration_results = None
right_calibration_results = None

# Calibrate each camera separately
for img_path_left, img_path_right in zip(images_left, images_right):
    img_left = cv.imread(img_path_left)
    img_right = cv.imread(img_path_right)
    gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
    gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret_left, corners_left = cv.findChessboardCorners(gray_left, chessboard_size, None)
    ret_right, corners_right = cv.findChessboardCorners(gray_right, chessboard_size, None)

    # If found, add object points, image points
    if ret_left and ret_right:
        objpoints.append(objp)
        corners2_left = cv.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        corners2_right = cv.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_left.append(corners2_left)
        imgpoints_right.append(corners2_right)

# Perform individual camera calibration for left and right cameras
left_calibration_results = cv.calibrateCamera(objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
right_calibration_results = cv.calibrateCamera(objpoints, imgpoints_right, gray_right.shape[::-1], None, None)


#save the induvidial results
left_filename = 'left_calibration_results.xml'
right_filename = 'right_calibration_results.xml'

# Open the files for writing
left_fs = cv.FileStorage(left_filename, cv.FILE_STORAGE_WRITE)
right_fs = cv.FileStorage(right_filename, cv.FILE_STORAGE_WRITE)

# Save left camera calibration results
left_fs.write("cameraMatrix1", left_calibration_results[1])
left_fs.write("distCoeffs1", left_calibration_results[2])

# Save right camera calibration results
right_fs.write("cameraMatrix2", right_calibration_results[1])
right_fs.write("distCoeffs2", right_calibration_results[2])

# Close the files
left_fs.release()
right_fs.release()


# Unpack the results
left_retval, cameraMatrix1, distCoeffs1, _, _ = left_calibration_results
right_retval, cameraMatrix2, distCoeffs2, _, _ = right_calibration_results

# Check if calibration was successful
if left_retval and right_retval:
    print("Camera calibration successful for both left and right cameras")

    # Perform stereo calibration using individual camera calibration results
    retval_stereo, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_left.shape[::-1],
        criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=cv.CALIB_FIX_INTRINSIC
    )

    if retval_stereo:
        print("Stereo calibration success of:", retval_stereo)
        print("Rotation matrix:\n", R)
        print("Translation vector:\n", T)

        # Save stereo calibration data
        filename = 'stereo_calibration_data.xml'  # Adjust the path as needed
        fs = cv.FileStorage(filename, cv.FILE_STORAGE_WRITE)

        fs.write("cameraMatrix1", cameraMatrix1)
        fs.write("distCoeffs1", distCoeffs1)
        fs.write("cameraMatrix2", cameraMatrix2)
        fs.write("distCoeffs2", distCoeffs2)
        fs.write("R", R)
        fs.write("T", T)
        fs.write("E", E)
        fs.write("F", F)

        fs.release()

        print(f"Stereo calibration data saved to {filename}")


        R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, gray_left.shape[::-1], R, T)

        # Initialize rectification maps for each camera
        map1x, map1y = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_left.shape[::-1], cv.CV_32FC1)
        map2x, map2y = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_right.shape[::-1], cv.CV_32FC1)

        # Rectify the last image pair
        rectified_img_left = cv.remap(img_left, map1x, map1y, cv.INTER_LINEAR)
        rectified_img_right = cv.remap(img_right, map2x, map2y, cv.INTER_LINEAR)

        # Display the rectified images
        cv.imshow("Rectified Left Image", rectified_img_left)
        cv.imshow("Rectified Right Image", rectified_img_right)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Stereo calibration failed.")
else:
    print("Camera calibration failed for one or both cameras.")
