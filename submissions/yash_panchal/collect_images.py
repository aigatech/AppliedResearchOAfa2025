# collect_images.py
import cv2
import os
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_images = 100
cap = cv2.VideoCapture(0)

# --- THIS IS THE ONLY LINE THAT CHANGES ---
# Collect all 24 static letters (A-Z, excluding J and Z)
letters = [chr(65 + i) for i in range(26) if chr(65 + i) not in ['J', 'Z']]

for letter in letters:
    class_dir = os.path.join(DATA_DIR, letter)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting images for class: {letter}')
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f'Ready for "{letter}"? Press "S" to start.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('s'):
            break

    print('Collecting...')
    counter = 0
    while counter < number_of_images:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        image_path = os.path.join(class_dir, f'{int(time.time() * 1000)}.jpg')
        cv2.imwrite(image_path, frame)
        counter += 1
        print(f'Collected image {counter}/{number_of_images} for {letter}')
    time.sleep(2)

cap.release()
cv2.destroyAllWindows()