# translator_custom.py
import cv2
from transformers import pipeline
from PIL import Image

print("Loading your custom ASL model...")
classifier = pipeline("image-classification", model="yashcpanchal/ASL_Alphabet_Classifier")
print("Model loaded successfully!")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    outputs = classifier(pil_image)
    prediction = outputs[0]
    label = prediction['label']
    score = prediction['score']

    text = f"{label} ({score:.2f})"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Custom ASL Translator', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()