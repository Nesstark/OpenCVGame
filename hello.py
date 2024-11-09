import cv2
import numpy as np
import random

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Game variables
rocket_x = 320
rocket_y = 400
rocket_speed = 5
obstacles = []
score = 0

def draw_rocket(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 20), (x + 10, y + 20), (0, 0, 255), -1)

def draw_obstacle(frame, x, y):
    cv2.rectangle(frame, (x - 10, y - 10), (x + 10, y + 10), (255, 0, 0), -1)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detect face and control rocket
    for (x, y, w, h) in faces:
        rocket_x = x + w // 2

    # Draw rocket
    draw_rocket(frame, rocket_x, rocket_y)

    # Generate obstacles
    if random.randint(1, 20) == 1:
        obstacles.append([random.randint(20, 620), 0])

    # Move obstacles
    for obstacle in obstacles:
        obstacle[1] += rocket_speed
        draw_obstacle(frame, obstacle[0], obstacle[1])
        if obstacle[1] > 480:
            obstacles.remove(obstacle)
            score += 1

    # Check for collisions
    for obstacle in obstacles:
        if abs(obstacle[0] - rocket_x) < 20 and abs(obstacle[1] - rocket_y) < 30:
            cv2.putText(frame, 'Game Over', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow('Rocket Launcher Game', frame)
            cv2.waitKey(3000)
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Display score
    cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow('Rocket Launcher Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()