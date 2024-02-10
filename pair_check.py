import cv2 as cv
import os
import glob

# Parameters
chessboard_size = (7, 9)  # Adjust based on your chessboard (number of inner corners per chessboard row and column)
images_left_path = 'images/Left/*.jpg'  # Adjust path as needed
images_right_path = 'images/Right/*.jpg'  # Adjust path as needed

# Load image file paths
images_left = sorted(glob.glob(images_left_path))
images_right = sorted(glob.glob(images_right_path))

# Ensure we have pairs of images
assert len(images_left) == len(images_right), "Mismatch in the number of left and right images"

def process_image_pairs(images_left, images_right, chessboard_size):
    for img_path_left, img_path_right in zip(images_left, images_right):
        img_left = cv.imread(img_path_left)
        img_right = cv.imread(img_path_right)

        # Convert to grayscale
        gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
        gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv.findChessboardCorners(gray_right, chessboard_size, None)

        # If found, draw corners
        if ret_left and ret_right:
            cv.drawChessboardCorners(img_left, chessboard_size, corners_left, ret_left)
            cv.drawChessboardCorners(img_right, chessboard_size, corners_right, ret_right)
            combined_img = cv.hconcat([img_left, img_right])  # Concatenate images for easier visualization
            
            # Show the images with corners
            cv.imshow('Stereo Pair with Detected Corners', combined_img)

            # User decides to keep or delete the image pair
            print(f"Inspecting image pair: {img_path_left} and {img_path_right}")
            key = cv.waitKey(0) & 0xFF
            if key == ord('d'):  # Press 'd' to delete
                os.remove(img_path_left)
                os.remove(img_path_right)
                print("Deleted the image pair.")
            elif key == ord('q'):  # Press 'q' to quit the program
                break
            else:
                print("Keeping the image pair.")
        else:
            os.remove(img_path_left)
            os.remove(img_path_right)
            print(f"Failed to detect chessboard corners in the pair: {img_path_left} and {img_path_right}")

    cv.destroyAllWindows()

process_image_pairs(images_left, images_right, chessboard_size)
