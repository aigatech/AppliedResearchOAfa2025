import cv2
import torch
import threading
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from transformers import DetrImageProcessor, DetrForObjectDetection, BlipProcessor, BlipForConditionalGeneration
from queue import Queue
import random

class RobotVisionUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Vision System ft. Good Old Fashioned Hate")
        
        # Initialize variables
        self.cap = None
        self.running = False
        self.webcam_thread = None
        self.frame_queue = Queue(maxsize=1)
        
        # Load models
        self.load_models()
        
        # Create UI
        self.create_ui()
        
        self.gt_messages = [
            "GO JACKETS!",
            "WHAT'S THE GOOD WORD?",
            "THWg!",
            "Honk if you love Buzz!",
            "Be ready to be stung!!",
            "BUZZ BUZZ!",
            "RAMBLIN' WRECK!"
        ]
        self.uga_messages = [
            "UGA ALERT! STAY AWAY!",
            "RED ALERT: UGA DETECTED!",
            "WARNING: BULLDOG IN VICINITY!",
            "EVACUATION RECOMMENDED: UGA PRESENT!",
            "DANGER: UGA SPOTTED!"
        ]
        
        # Cooldown for alerts
        self.last_alert_time = 0
        self.alert_cooldown = 2.0  # seconds
        
    def load_models(self):
        print("Loading DETR...")
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        print("Loading BLIP...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def create_ui(self):
        # Create main container
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create buttons frame
        self.button_frame = ttk.Frame(self.main_container)
        self.button_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Add start button
        self.start_btn = ttk.Button(self.button_frame, text="Start Camera 🎥", 
                                    command=self.start_webcam_mode)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        # Create display canvas
        self.canvas = tk.Canvas(self.main_container, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create status label
        self.status_label = ttk.Label(self.main_container, text="Status: Ready 🐝")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
    def check_color(self, frame, color):
        """Check if a significant amount of a specific color is present in the frame."""
        # Convert frame to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if color == "red":
            # Red has two ranges in HSV
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
        elif color == "yellow":
            # Expanded yellow/gold range for GT colors
            lower_yellow = np.array([15, 50, 50])    # Lowered saturation and value thresholds
            upper_yellow = np.array([45, 255, 255])  # Increased range to catch more yellow/gold
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
        # Apply some morphological operations to reduce noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Different thresholds for different colors
        threshold = 0.03 if color == "yellow" else 0.05  # Lower threshold for yellow/gold
        
        # Calculate percentage of color in frame
        color_percentage = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1])
        return color_percentage > threshold

    def start_webcam_mode(self):
        """Start webcam mode."""
        if self.running:
            return
            
        self.running = True
        
        # Initialize webcam if needed
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            
        # Start webcam thread
        self.webcam_thread = threading.Thread(target=self.webcam_loop)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()
        
        self.status_label.config(text="Status: Webcam Mode Active 🎥")
        
    def webcam_loop(self):
        """Main webcam processing loop."""
        self.status_label.config(text="Status: Running 🎥")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process frame for GT vs UGA detection
            processed_frame = self.process_frame(frame)
            
            # Convert to PIL format for display
            image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep reference
            
            # Update UI
            self.root.update()
            
            # Convert to PIL format for display
            image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas
            self.canvas.config(width=image.width, height=image.height)
            self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            self.canvas.image = photo  # Keep reference
            
            # Update UI
            self.root.update()
            
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Process a frame for object detection and GT vs UGA detection."""
        frame = frame.copy()
        current_time = time.time()
        
        # Set robot's position at the bottom center of the frame
        h, w, _ = frame.shape
        robot_position = (w // 2, h - 50)
        
        # Run DETR detection
        small_frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        inputs = self.detr_processor(images=rgb_frame, return_tensors="pt")
        outputs = self.detr_model(**inputs)
        logits = outputs.logits[0]
        boxes = outputs.pred_boxes[0]
        
        # Process detected objects
        probas = logits.softmax(-1)
        keep = probas.max(-1).values > 0.9
        
        # Check for red (UGA) and yellow (GT) colors
        has_red = self.check_color(frame, "red")
        has_yellow = self.check_color(frame, "yellow")
        
        # Show color alerts if enough time has passed
        if current_time - self.last_alert_time >= self.alert_cooldown:
            if has_red:
                # Show UGA alert
                self.last_alert_time = current_time
                message = random.choice(self.uga_messages)
                self.root.after(100, lambda: messagebox.showwarning("ALERT!", message))
            elif has_yellow:
                # Show GT celebration
                self.last_alert_time = current_time
                message = random.choice(self.gt_messages)
                self.root.after(100, lambda: messagebox.showinfo("GO JACKETS!", message))
        
        # Draw detected objects and trajectories
        for logit, box in zip(logits[keep], boxes[keep]):
            cls_id = logit.argmax().item()
            if cls_id == self.detr_model.config.num_labels:
                continue
            label = self.detr_model.config.id2label[cls_id]
            score = logit.softmax(-1).max().item()
            cx, cy, bw, bh = box
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            
            # Calculate object center
            object_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            # Generate and draw trajectory
            x = np.linspace(robot_position[0], object_center[0], 20)
            y = np.linspace(robot_position[1], object_center[1], 20)
            y = y - np.sin(np.linspace(0, np.pi, 20)) * 30
            trajectory = np.column_stack((x, y)).astype(np.int32)
            
            # Draw trajectory as a curved line
            for i in range(len(trajectory) - 1):
                cv2.line(frame, tuple(trajectory[i]), tuple(trajectory[i + 1]), 
                        (255, 0, 0), 2)
            
            # Draw bounding box and label
            box_color = (0, 0, 255) if has_red else (0, 255, 255) if has_yellow else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
            # Draw robot position and target position
            cv2.circle(frame, robot_position, 5, (0, 255, 0), -1)  # Robot position
            cv2.circle(frame, object_center, 5, box_color, -1)  # Target position
        
        # Add color detection labels
        if has_red:
            cv2.putText(frame, "UGA DETECTED!", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if has_yellow:
            cv2.putText(frame, "GO JACKETS!", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        return frame


if __name__ == "__main__":
    root = tk.Tk()
    app = RobotVisionUI(root)
    root.mainloop()
