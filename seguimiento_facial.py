import cv2
print("OpenCV version:", cv2.__version__)

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the camera (adjust for Mac with AVFoundation if needed)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not access the camera.")
        break

    # Flip horizontally for mirror mode
    frame = cv2.flip(frame, 1)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Take the first detected face
        (x, y, w, h) = faces[0]

        # Expand the crop for zoom (with padding)
        padding = 60
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        # Crop the region and zoom (rescale)
        face_zoom = frame[y1:y2, x1:x2]
        frame = cv2.resize(face_zoom, (frame.shape[1], frame.shape[0]))

    # Display the result
    cv2.imshow("Face Tracking with Zoom", frame)

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
