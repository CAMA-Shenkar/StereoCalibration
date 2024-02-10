import cv2

# Function to be called whenever trackbar position is changed
def on_trackbar_focus(val):
    cap.set(cv2.CAP_PROP_FOCUS, val)

# Initialize camera
cap = cv2.VideoCapture(1) # Adjust the device index as necessary
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Turn off autofocus
cap.set(3, 1920)

# Create a window
cv2.namedWindow('Camera')

# Create trackbar for focus adjustment
# Arguments: trackbar name, window name, value range(min, max), callback function
max_focus = 255 # This value may vary between cameras; you might need to adjust it
cv2.createTrackbar('Focus', 'Camera', 0, max_focus, on_trackbar_focus)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
