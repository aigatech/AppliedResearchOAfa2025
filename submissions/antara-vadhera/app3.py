import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime, timedelta
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c) 

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

hosts_path = r"C:\Windows\System32\drivers\etc\hosts"  #windows, idk for mac/linux yet
redirect = "127.0.0.1"
websites = ["www.youtube.com", "youtube.com", "www.instagram.com", "instagram.com"]

def block_websites():
    with open(hosts_path, 'r+') as file:
        content = file.read()
        for site in websites:
            if site not in content:
                file.write(f"{redirect} {site}\n")
    print("Websites blocked.")

def unblock_websites():
    with open(hosts_path, 'r+') as file:
        lines = file.readlines()
        file.seek(0)
        for line in lines:
            if not any(site in line for site in websites):
                file.write(line)
        file.truncate()
    print("Websites unblocked.")

# initial
block_websites()

min = 5
min_per_pushup = 1
counter = 0
stage = None

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        landmarks = results.pose_landmarks.landmark

        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        angle = calculate_angle(shoulder, elbow, wrist)

        # Push-up logic
        if angle > 160:
            stage = "up"
        if angle < 90 and stage == "up":
            stage = "down"
            counter += 1
            print(f"Push-up count: {counter}")

        # add landmark for torso and time checking for a proper push-up. currently easy to cheat but oh well. 

        # drawing
        cv2.putText(image, str(int(angle)),
                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    except:
        pass

    # Render push-up counter
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(image, 'Push-ups', (15,12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter),
                (10,60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # Draw landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)



    if counter >= min:
        # Calculate unlock time based on push-ups
        scroll_minutes = counter * min_per_pushup
        print(f"You did {counter} push-ups! Unlocking websites for {scroll_minutes} minutes.")
        unblock_websites()

        end_time = datetime.now() + timedelta(minutes=scroll_minutes)
        while datetime.now() < end_time:
            time.sleep(5)

        block_websites()
        print("Time's up! Websites blocked again.")
        counter = 0  # reset for next session
        print("Start doing push-ups again!")

    cv2.imshow('Push-up Counter', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
