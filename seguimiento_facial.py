import cv2

print("OpenCV version:", cv2.__version__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open camera (AVFoundation backend for macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# Initialize previous face position
prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
alpha = 0.2  # smoothing factor (0 = no update, 1 = instant change)

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
        (x, y, w, h) = faces[0]

        # Smooth the detection using previous coordinates
        prev_x = int(prev_x * (1 - alpha) + x * alpha)
        prev_y = int(prev_y * (1 - alpha) + y * alpha)
        prev_w = int(prev_w * (1 - alpha) + w * alpha)
        prev_h = int(prev_h * (1 - alpha) + h * alpha)

        # Add padding around the face
        padding = 60
        padding_top = 80
        padding_bottom = 140

        x1 = max(prev_x - padding, 0)
        y1 = max(prev_y - padding_top, 0)
        x2 = min(prev_x + prev_w + padding, frame.shape[1])
        y2 = min(prev_y + prev_h + padding_bottom, frame.shape[0])

        # Zoom into the face region
        face_zoom = frame[y1:y2, x1:x2]
        frame = cv2.resize(face_zoom, (frame.shape[1], frame.shape[0]))

    # Show the frame
    cv2.imshow("Stable Face Zoom", frame)

    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
