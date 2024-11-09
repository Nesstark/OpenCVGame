import cv2
import numpy as np
import random
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
initial_speed = 11200  # Rocket speed in m/s for low Earth orbit approximation
score = 0
target_altitude = 10_000_000  # Exosphere height in meters

def draw_rocket(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 20), (x + 10, y + 20), (0, 0, 255), -1)

def draw_obstacle(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), -1)

def draw_progress_bar(frame, score, max_height=None):
    """Draws a vertical progress bar on the right side of the frame based on the score."""
    if max_height is None:
        max_height = frame.shape[0]  # Set max_height to frame height if not provided
    bar_height = int((score / target_altitude) * max_height)  # Scale bar height to target altitude
    bar_y = frame.shape[0] - bar_height  # Calculate starting y-position
    bar_x = frame.shape[1] - 30  # Position on the right side of the screen
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, frame.shape[0]), (255, 255, 255), -1)

def display_score(frame, score):
    """Displays the score with a black background in the top-left corner."""
    altitude_text = f'Altitude: {score} m'
    (text_width, text_height), _ = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (10 + text_width + 20, 10 + text_height + 20), (0, 0, 0), -1)  # Black background
    cv2.putText(frame, altitude_text, (20, 30 + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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

    # Calculate score based on time elapsed, in meters
    elapsed_time = time.time() - start_time
    score = int(initial_speed * elapsed_time)  # Altitude in meters

    # Check for win condition
    if score >= target_altitude:
        cv2.putText(frame, 'You Win!', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow('Rocket Launcher Game', frame)
        cv2.waitKey(3000)
        break

    # Dynamically increase difficulty
    spawn_chance = max(5, 25 - score // 100000)  # Reduce spawn chance for higher altitude
    if random.randint(1, spawn_chance) == 1:
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

    # Display altitude with a black background
    display_score(frame, score)

    # Draw progress bar representing distance traveled into space
    draw_progress_bar(frame, score)

    # Show frame
    cv2.imshow('Rocket Launcher Game', frame)

    # Add kill switch for the "Q" key to quit the game
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Game exited by pressing 'Q'.")
        break

cap.release()
cv2.destroyAllWindows()