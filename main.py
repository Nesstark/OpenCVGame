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
rocket_x = 320  # Initial rocket position
rocket_y = 400  # Initial rocket position
rocket_speed = 5    # Rocket speed in pixels per frame
obstacles = []  # List of obstacles [x, y, type]
smoothing_factor = 0.2  # Adjust for smoother face-following movement
start_time = time.time()  # Record the start time for score calculation
initial_speed = 11200  # Rocket speed in m/s for low Earth orbit approximation
score = 0   # Altitude in meters
target_altitude = 10_000_000  # Exosphere height in meters

# Function to draw rocket
def draw_rocket(frame, x, y):
    # Simple rocket design (pointed top and body)
    cv2.polylines(frame, [np.array([[x, y - 20], [x - 10, y], [x + 10, y]], np.int32)], isClosed=True, color=(0, 0, 255), thickness=3)  # Rocket top
    cv2.rectangle(frame, (x - 10, y), (x + 10, y + 20), (0, 0, 255), -1)  # Rocket body
    # Rocket flame with more dynamic flame-like appearance
    flame_points = np.array([
        [x, y + 20], [x - 4, y + 30], [x - 8, y + 40], [x - 4, y + 50],
        [x, y + 40], [x + 4, y + 50], [x + 8, y + 40], [x + 4, y + 30],
        [x - 2, y + 45], [x + 2, y + 45], [x - 6, y + 35], [x + 6, y + 35]
    ], np.int32)
    cv2.fillPoly(frame, [flame_points], color=(0, 165, 255))  # Orange flame

# Function to draw asteroid
def draw_asteroid(frame, x, y):
    # Larger irregular shaped gray asteroid with filled color
    points = np.array([[x, y], [x + 30, y - 10], [x + 10, y + 20], [x - 20, y + 16]], np.int32)
    cv2.fillPoly(frame, [points], color=(105, 105, 105))  # Darker gray fill
    cv2.polylines(frame, [points], isClosed=True, color=(105, 105, 105), thickness=3)  # Same color border

# Function to draw satellite
def draw_satellite(frame, x, y):
    # Rectangular satellite with solar panels and an antenna
    cv2.rectangle(frame, (x - 12, y - 12), (x + 12, y + 12), (192, 192, 192), -1)  # Satellite body
    cv2.rectangle(frame, (x - 30, y - 8), (x - 12, y + 8), (255, 255, 255), -1)  # Left solar panel
    cv2.rectangle(frame, (x + 12, y - 8), (x + 30, y + 8), (255, 255, 255), -1)  # Right solar panel
    cv2.line(frame, (x, y - 12), (x, y - 30), (0, 0, 0), 2)  # Antenna

# Function to draw progress bar
def draw_progress_bar(frame, score, max_height=None):
    """Draws a vertical progress bar on the right side of the frame based on the score."""
    if max_height is None:
        max_height = frame.shape[0]  # Set max_height to frame height if not provided
    bar_height = int((score / target_altitude) * max_height)  # Scale bar height to target altitude
    bar_y = frame.shape[0] - bar_height  # Calculate starting y-position
    bar_x = frame.shape[1] - 30  # Position on the right side of the screen

    # Determine the color based on the altitude
    if score < 12_000:
        color = (255, 255, 255)  # White for troposphere
    elif score < 50_000:
        color = (255, 200, 200)  # Light blue for stratosphere
    elif score < 80_000:
        color = (255, 150, 150)  # Medium blue for mesosphere
    elif score < 700_000:
        color = (255, 100, 100)  # Darker blue for thermosphere
    elif score < 1_000_000:
        color = (255, 50, 50)  # Even darker blue for exosphere
    elif score < 5_000_000:
        color = (255, 25, 25)  # Very dark blue for higher exosphere
    elif score < 8_000_000:
        color = (255, 10, 10)  # Near black for even higher exosphere
    else:
        color = (255, 0, 0)  # Black for near space

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, frame.shape[0]), color, -1)

# Function to display score
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
        cv2.putText(frame, 'Mission Complete!', (frame_width // 2 - 120, frame_height // 2 - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Display the final altitude (how far you came)
        altitude_text = f'Altitude: {score} m'
        (text_width, text_height), _ = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        box_margin = 20
        cv2.rectangle(frame, (frame_width // 2 - text_width // 2 - box_margin, frame_height // 2 + 10), 
                      (frame_width // 2 + text_width // 2 + box_margin, frame_height // 2 + 10 + text_height + 2 * box_margin), (0, 0, 0), -1)
        cv2.putText(frame, altitude_text, (frame_width // 2 - text_width // 2, frame_height // 2 + 10 + text_height + box_margin), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the final score and wait before exiting
        cv2.imshow('Rocket Launcher Game', frame)
        cv2.waitKey(3000)
        break

    # Dynamically increase difficulty
    spawn_chance = max(1, 20 - score // 50000)  # Reduce spawn chance for higher altitude, making it harder earlier
    if random.randint(1, spawn_chance) <= 2:  # Increase the likelihood of spawning obstacles
        obstacle_type = random.choice(["asteroid", "satellite"])  # Randomly choose between asteroid or satellite
        obstacles.append([random.randint(20, frame_width - 20), 0, obstacle_type])

    # Move and draw obstacles
    obstacles_to_remove = []
    for obstacle in obstacles:
        obstacle[1] += rocket_speed
        if obstacle[2] == "asteroid":
            draw_asteroid(frame, obstacle[0], obstacle[1])
        else:
            draw_satellite(frame, obstacle[0], obstacle[1])
        if obstacle[1] > frame_height:
            obstacles_to_remove.append(obstacle)

    # Remove marked obstacles
    for obstacle in obstacles_to_remove:
        obstacles.remove(obstacle)

    # Check for collisions
    for obstacle in obstacles:
        if abs(obstacle[0] - rocket_x) < 20 and abs(obstacle[1] - rocket_y) < 30:
            # Game Over
            game_over_text = 'Mission Over'
            (go_text_width, go_text_height), _ = cv2.getTextSize(game_over_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
            cv2.putText(frame, game_over_text, (frame_width // 2 - go_text_width // 2, frame_height // 2 - go_text_height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
             # Display the final altitude (how far you came)
            altitude_text = f'Altitude: {score} m'
            (text_width, text_height), _ = cv2.getTextSize(altitude_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (frame_width // 2 - text_width // 2 - 10, frame_height // 2 + 10), 
                          (frame_width // 2 + text_width // 2 + 10, frame_height // 2 + 10 + text_height + 40), (0, 0, 0), -1)
            cv2.putText(frame, altitude_text, (frame_width // 2 - text_width // 2, frame_height // 2 + text_height + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the final score and wait before exiting
            cv2.imshow('Rocket Launcher Game', frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Draw progress bar and display score
    draw_progress_bar(frame, score)
    display_score(frame, score)

    # Show the frame
    cv2.imshow('Rocket Launcher Game', frame)

    # Add kill switch for the "Q" key to quit the game
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Game exited by pressing 'Q'.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()