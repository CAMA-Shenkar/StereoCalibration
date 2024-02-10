import cv2 as cv

def open_camera(index=0):
    # Try to open the video capture device
    cap = cv.VideoCapture(index)
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
    
    if not cap.isOpened():
        print(f"Error: Could not open video capture device at index {index}")
        return

    print(f"Opening camera at index {index}...")

    # Set window name
    window_name = f"Camera Index {index}"

    # Loop to continuously fetch frames
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        cv.imshow(window_name, frame)

        # Break the loop when 'q' is pressed
        if cv.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture and close any open windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    open_camera(1)  # Change the index here if needed
    
