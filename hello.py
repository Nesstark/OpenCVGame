import cv2
import numpy as np
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize the video capture
cap = cv2.VideoCapture(0)
# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Game variables
rocket_x = 320
rocket_y = 400
rocket_speed = 5
obstacles = []
smoothing_factor = 0.2  # Adjust for smoother face-following movement
start_time = time.time()  # Record the start time for score calculation

def draw_rocket(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 20), (x + 10, y + 20), (0, 0, 255), -1)

def draw_obstacle(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), -1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe requires RGB input

    # Perform face detection
    results = face_detection.process(rgb_frame)
    if results.detections:
        # Get the first detected face
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_center_x = int((bboxC.xmin + bboxC.width / 2) * frame_width)

            # Smooth movement of the rocket towards the face position
            rocket_x = int(rocket_x + (face_center_x - rocket_x) * smoothing_factor)
            break  # Use the first face detected

    # Draw the rocket
    draw_rocket(frame, rocket_x, rocket_y)

    # Generate obstacles across the full frame width
    if random.randint(1, 25) == 1:
        obstacles.append([random.randint(20, frame_width - 20), 0])

    # Move and draw obstacles
    obstacles_to_remove = []
    for obstacle in obstacles:
        obstacle[1] += rocket_speed
        draw_obstacle(frame, obstacle[0], obstacle[1])
        if obstacle[1] > frame_height:
            obstacles_to_remove.append(obstacle)

    # Remove marked obstacles
    for obstacle in obstacles_to_remove:
        obstacles.remove(obstacle)

    # Check for collisions
    for obstacle in obstacles:
        if abs(obstacle[0] - rocket_x) < 20 and abs(obstacle[1] - rocket_y) < 30:
            cv2.putText(frame, 'Game Over', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Rocket Launcher Game', frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Calculate the score based on survival time
    elapsed_time = int(time.time() - start_time)
    score = elapsed_time * 10  # Increase score based on time survived

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Rocket Launcher Game', frame)

    # Add kill switch for the "Q" key to quit the game
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Game exited by pressing 'Q'.")
        break

cap.release()
cv2.destroyAllWindows()