import cv2
import dlib
from deepface import DeepFace

# Load the pre-trained face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open a video file or capture device
video_path = 0  # 0 for webcam, or replace with video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}.")
    exit()

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # If no faces are detected, continue to the next frame
    if len(faces) == 0:
        cv2.putText(frame, "No faces detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # Loop through all detected faces
        for face in faces:
            # Get the landmarks of the face
            landmarks = predictor(gray, face)

            # Draw landmarks on the face
            for n in range(0, 68):  # Loop through all 68 landmarks
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green dots for landmarks

            # Extract the region of interest (face) from the frame
            (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Predict emotion using DeepFace
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                predicted_emotion = result[0]['dominant_emotion']
            except Exception as e:
                print(f"Error in emotion analysis: {e}.")
                predicted_emotion = "No Prediction"

            # Display predicted emotion
            cv2.putText(frame, f"Emotion: {predicted_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame with landmarks and predicted emotion
    cv2.imshow("Landmarks and Emotion Prediction", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
cap.release()
cv2.destroyAllWindows()
