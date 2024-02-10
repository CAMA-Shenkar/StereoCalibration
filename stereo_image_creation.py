import cv2
import os
from datetime import datetime

# Function to ensure the directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure image directories exist
ensure_dir('images/Right')
ensure_dir('images/Left')

# Open the cameras
cap_right = cv2.VideoCapture(2)  # Index 1 for the right camera
cap_left = cv2.VideoCapture(1)  # Index 0 for the left camera

cap_left.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap_right.set(cv2.CAP_PROP_AUTOFOCUS, 0)

cap_left.set(cv2.CAP_PROP_FOCUS, 50)
cap_right.set(cv2.CAP_PROP_FOCUS, 50)

# cap_left.set(3, 1920)
# cap_right.set(3, 1920)


if not cap_right.isOpened() or not cap_left.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

while True:
    # Capture frame-by-frame from each camera
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()

    # Check if frames are successfully read
    if not ret_right or not ret_left:
        print("Error: Can't receive frame from one or both cameras. Exiting ...")
        break

    # Display the frames
    cv2.imshow('Right Camera POV', frame_right)
    cv2.imshow('Left Camera POV', frame_left)

    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # If 's' is pressed, save both frames
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        right_filename = f"images/Right/{timestamp}.jpg"
        left_filename = f"images/Left/{timestamp}.jpg"
        
        cv2.imwrite(right_filename, frame_right)
        cv2.imwrite(left_filename, frame_left)

        print(f"Saved Right Image to {right_filename}")
        print(f"Saved Left Image to {left_filename}")

    # Break the loop on 'q' press
    elif key == ord('q'):
        break

# When everything is done, release the captures and close windows
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
