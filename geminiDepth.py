import cv2
import numpy as np

def load_calibration_data(calibration_file):
    # Load stereo calibration data from XML file
    calibration_data = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    
    # Load calibration parameters
    cameraMatrix1 = calibration_data.getNode("cameraMatrix1").mat()
    distCoeffs1 = calibration_data.getNode("distCoeffs1").mat()
    cameraMatrix2 = calibration_data.getNode("cameraMatrix2").mat()
    distCoeffs2 = calibration_data.getNode("distCoeffs2").mat()
    R = calibration_data.getNode("R").mat()
    T = calibration_data.getNode("T").mat()
    E = calibration_data.getNode("E").mat()
    F = calibration_data.getNode("F").mat()
    
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

def rectify_images(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    
    # Create a Brute Force matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_LMEDS)
    
    # Rectify images
    h1, w1 = img1_gray.shape
    h2, w2 = img2_gray.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(src_pts, dst_pts, F, (w1, h1))
    
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
    
    return img1_rectified, img2_rectified





def compute_depth_map(img1_rectified, img2_rectified):
    # Convert rectified images to grayscale
    img1_gray = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    
    # Compute depth map
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1_gray, img2_gray)
    
    # Normalize the disparity map for better visualization
    min_disp = disparity.min()
    max_disp = disparity.max()
    disparity = np.uint8(255 * (disparity - min_disp) / (max_disp - min_disp))
    
    return disparity

def main():
    # Load stereo calibration data
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, _, _ = load_calibration_data("stereo_calibration_data.xml")
    
    # Open camera feeds
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    while True:
        ret1, img1 = cap1.read()
        ret2, img2 = cap2.read()
        
        if not ret1 or not ret2:
            print("Error: Unable to capture frames")
            break
        
        # Rectify images
        img1_rectified, img2_rectified = rectify_images(img1, img2)
        
        # Compute depth map
        depth_map = compute_depth_map(img1_rectified, img2_rectified)
        
        # Display images and depth map
        cv2.imshow("Left Image", img1_rectified)
        cv2.imshow("Right Image", img2_rectified)
        cv2.imshow("Depth Map", depth_map)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
