import cv2
import numpy as np
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize game variables
width, height = 640, 480
ball_pos = [width // 2, height // 2]
ball_vel = [4, 4]
paddle1_pos = height // 2
paddle2_pos = height // 2
paddle_width, paddle_height = 10, 100
score1, score2 = 0, 0

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

def detect_hands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    return result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    result = detect_hands(frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * height)
            if x < width // 2:
                paddle1_pos = y
            else:
                paddle2_pos = y

    # Update ball position
    ball_pos[0] += ball_vel[0]
    ball_pos[1] += ball_vel[1]

    # Ball collision with top and bottom
    if ball_pos[1] <= 0 or ball_pos[1] >= height:
        ball_vel[1] = -ball_vel[1]

    # Ball collision with paddles
    if (ball_pos[0] <= paddle_width and paddle1_pos - paddle_height // 2 <= ball_pos[1] <= paddle1_pos + paddle_height // 2) or \
       (ball_pos[0] >= width - paddle_width and paddle2_pos - paddle_height // 2 <= ball_pos[1] <= paddle2_pos + paddle_height // 2):
        ball_vel[0] = -ball_vel[0]

    # Ball out of bounds
    if ball_pos[0] <= 0:
        score2 += 1
        ball_pos = [width // 2, height // 2]
    elif ball_pos[0] >= width:
        score1 += 1
        ball_pos = [width // 2, height // 2]

    # Draw everything
    frame = cv2.rectangle(frame, (0, paddle1_pos - paddle_height // 2), (paddle_width, paddle1_pos + paddle_height // 2), (255, 0, 0), -1)
    frame = cv2.rectangle(frame, (width - paddle_width, paddle2_pos - paddle_height // 2), (width, paddle2_pos + paddle_height // 2), (0, 0, 255), -1)
    frame = cv2.circle(frame, tuple(ball_pos), 10, (0, 255, 0), -1)
    frame = cv2.putText(frame, f'Score: {score1} - {score2}', (width // 2 - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Pong Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()