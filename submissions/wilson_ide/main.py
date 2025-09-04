import torch
import numpy as np
import argparse
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mediapipe as mp

from transformers import AutoProcessor, YolosForObjectDetection

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# COCO keypoint names for reference
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Define connections between keypoints for visualization
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

def load_models():
    """Load object detection model and initialize MediaPipe Pose"""
    # Object detection model for finding humans - using YOLOS which is more reliable
    detector_processor = AutoProcessor.from_pretrained("hustvl/yolos-tiny")
    detector_model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
    detector_model.to(device)
    
    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    
    return detector_model, detector_processor, pose

def detect_humans(image, detector_model, detector_processor):
    """Detect humans in the image using YOLOS"""
    inputs = detector_processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = detector_model(**inputs)
    
    # Post-process to get detections
    target_sizes = torch.tensor([image.size[::-1]])
    results = detector_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.5
    )[0]
    
    human_boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # Look for a person (label 1) condfidence over 0.5
        if label == 1 and score > 0.5:
            human_boxes.append(box.cpu().numpy())
    
    return human_boxes

def estimate_pose(image, human_box, pose):
    """Estimate pose for a detected human using MediaPipe"""
    # Crop the human from the image
    x1, y1, x2, y2 = map(int, human_box)
    human_crop = image.crop((x1, y1, x2, y2))
    
    # Convert PIL image to numpy array
    img_array = np.array(human_crop)
    
    # Convert RGB to BGR for MediaPipe
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Process with MediaPipe
    results = pose.process(img_bgr)
    
    if not results.pose_landmarks:
        return []
    
    # Convert MediaPipe landmarks to our keypoint format
    landmarks = results.pose_landmarks.landmark
    keypoints = []
    
    # MediaPipe pose landmarks mapping to COCO format
    mp_to_coco = {
        mp_pose.PoseLandmark.NOSE: 0,
        mp_pose.PoseLandmark.LEFT_EYE: 1,
        mp_pose.PoseLandmark.RIGHT_EYE: 2,
        mp_pose.PoseLandmark.LEFT_EAR: 3,
        mp_pose.PoseLandmark.RIGHT_EAR: 4,
        mp_pose.PoseLandmark.LEFT_SHOULDER: 5,
        mp_pose.PoseLandmark.RIGHT_SHOULDER: 6,
        mp_pose.PoseLandmark.LEFT_ELBOW: 7,
        mp_pose.PoseLandmark.RIGHT_ELBOW: 8,
        mp_pose.PoseLandmark.LEFT_WRIST: 9,
        mp_pose.PoseLandmark.RIGHT_WRIST: 10,
        mp_pose.PoseLandmark.LEFT_HIP: 11,
        mp_pose.PoseLandmark.RIGHT_HIP: 12,
        mp_pose.PoseLandmark.LEFT_KNEE: 13,
        mp_pose.PoseLandmark.RIGHT_KNEE: 14,
        mp_pose.PoseLandmark.LEFT_ANKLE: 15,
        mp_pose.PoseLandmark.RIGHT_ANKLE: 16,
    }
    
    for mp_landmark, coco_idx in mp_to_coco.items():
        landmark = landmarks[mp_landmark]
        if landmark.visibility > 0.5:
            # Convert normalized coordinates to pixel coordinates
            x = landmark.x * (x2 - x1) + x1
            y = landmark.y * (y2 - y1) + y1
            keypoints.append((coco_idx, np.array([x, y]), landmark.visibility))
        else:
            keypoints.append((coco_idx, None, landmark.visibility))
    
    return keypoints

def classify_stance(keypoints):
    """Classify human stance as active or passive based on keypoint positions"""
    if not keypoints:
        return "unknown"
    
    # Extract keypoint indices and positions
    kp_dict = {idx: pos for idx, pos, score in keypoints if pos is not None}
    
    # Get key body parts
    left_shoulder = kp_dict.get(5)  # left shoulder
    right_shoulder = kp_dict.get(6)  # right shoulder
    left_elbow = kp_dict.get(7)  # left elbow
    right_elbow = kp_dict.get(8)  # right elbow
    left_wrist = kp_dict.get(9)  # left wrist
    right_wrist = kp_dict.get(10)  # right wrist
    left_hip = kp_dict.get(11)  # left hip
    right_hip = kp_dict.get(12)  # right hip
    left_knee = kp_dict.get(13)  # left knee
    right_knee = kp_dict.get(14)  # right knee
    left_ankle = kp_dict.get(15)  # left ankle
    right_ankle = kp_dict.get(16)  # right ankle
    nose = kp_dict.get(0)  # nose
    
    active_indicators = 0
    total_checks = 0
    
    # Heuristic 1: Arm positions (active if arms are raised or extended)
    h1 = 0
    total_arms = 0
    
    # Check left arm
    if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
        total_arms += 1
        # Check if arm is raised above shoulder level
        if left_wrist[1] < left_shoulder[1] - 20:  # wrist above shoulder
            active_indicators += 1
            h1 += 1
        # Check if arm is extended forward
        arm_length = np.linalg.norm(left_elbow - left_shoulder)
        forearm_length = np.linalg.norm(left_wrist - left_elbow)
        if forearm_length > arm_length * 0.8:  # extended arm
            active_indicators += 1
            h1 += 1
    
    # Check right arm
    if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
        total_arms += 1
        # Check if arm is raised above shoulder level
        if right_wrist[1] < right_shoulder[1] - 20:  # wrist above shoulder
            active_indicators += 1
            h1 += 1
        # Check if arm is extended forward
        arm_length = np.linalg.norm(right_elbow - right_shoulder)
        forearm_length = np.linalg.norm(right_wrist - right_elbow)
        if forearm_length > arm_length * 0.8:  # extended arm
            active_indicators += 1
            h1 += 1
    
    if total_arms > 0:
        total_checks += 1
    
    # Heuristic 2: Stance width (active if legs are spread apart)
    h2 = 0
    stance_width = 0
    if left_hip is not None and right_hip is not None:
        stance_width = abs(right_hip[0] - left_hip[0])
        if stance_width > 100:
            active_indicators += 1
            h2 += 1
        total_checks += 1
    
    # Heuristic 3: Forward lean (active if leaning forward)
    h3 = 0
    if nose is not None and left_hip is not None and right_hip is not None:
        hip_center = (left_hip + right_hip) / 2
        # Check if head is forward of hip center
        if nose[0] > hip_center[0] + 10:
            active_indicators += 1
            h3 += 1
        total_checks += 1
    
    # Heuristic 4: Leg Asymmetry (NEW - active if legs have different behavior)
    h4 = 0
    if (left_hip is not None and right_hip is not None and 
        left_knee is not None and right_knee is not None and
        left_ankle is not None and right_ankle is not None):
        
        # Calculate leg angles and positions
        # Left leg analysis
        left_hip_to_knee = np.linalg.norm(left_knee - left_hip)
        left_knee_to_ankle = np.linalg.norm(left_ankle - left_knee)
        left_leg_length = left_hip_to_knee + left_knee_to_ankle
        
        # Right leg analysis  
        right_hip_to_knee = np.linalg.norm(right_knee - right_hip)
        right_knee_to_ankle = np.linalg.norm(right_ankle - right_knee)
        right_leg_length = right_hip_to_knee + right_knee_to_ankle
        
        # Calculate knee bend ratios (lower = more bent)
        left_knee_bend_ratio = left_hip_to_knee / max(left_leg_length, 1)
        right_knee_bend_ratio = right_hip_to_knee / max(right_leg_length, 1)
        
        # Calculate ankle spread (how far apart the feet are)
        ankle_spread = abs(right_ankle[0] - left_ankle[0])
        knee_spread = abs(right_knee[0] - left_knee[0])
        
        # Check for asymmetric leg behavior
        knee_bend_diff = abs(left_knee_bend_ratio - right_knee_bend_ratio)
        ankle_knee_diff = abs(ankle_spread - knee_spread)
        
        # Active indicators for leg asymmetry:
        # 1. Different knee bend ratios (one leg more bent than other)
        if knee_bend_diff > 0.15:  # Significant difference in knee bend
            active_indicators += 1
            h4 += 1
        
        # 2. Ankles more spread than knees (running/walking stance)
        if ankle_spread > knee_spread * 1.3:  # Feet wider than knees
            active_indicators += 1
            h4 += 1
        
        # 3. One leg significantly more forward than the other
        left_leg_forward = left_ankle[0] - left_hip[0]
        right_leg_forward = right_ankle[0] - right_hip[0]
        leg_forward_diff = abs(left_leg_forward - right_leg_forward)
        
        if leg_forward_diff > 30:  # One leg significantly more forward
            active_indicators += 1
            h4 += 1
        
        # 4. Different vertical positions of knees (one up, one down)
        knee_height_diff = abs(left_knee[1] - right_knee[1])
        if knee_height_diff > 25:  # Significant height difference
            active_indicators += 1
            h4 += 1
        
        total_checks += 1
    
    # Classification logic
    if total_checks == 0:
        return "unknown"
    
    # Calculate activity score
    activity_score = active_indicators / total_checks
    
    print(f"heuristic 1 (arms): {h1}, heuristic 2 (stance): {h2}, heuristic 3 (lean): {h3}, heuristic 4 (legs): {h4}")
    print(f"active_indicators: {active_indicators}, total_checks: {total_checks}, activity_score: {activity_score:.2f}")
    
    if activity_score > 0.6:  # More conservative threshold
        return "active"
    else:
        return "passive"

def visualize_results(image, human_boxes, all_keypoints, stances):
    """Visualize the results with keypoints and stance classification"""
    # Convert PIL image to numpy array for OpenCV
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Draw each detected human
    for i, (box, keypoints, stance) in enumerate(zip(human_boxes, all_keypoints, stances)):
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        color = (0, 100, 255) if stance == "active" else (0, 255, 0)  # Orange for active, green for passive
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        
        # Draw stance label
        label = f"{stance.upper()}"
        cv2.putText(img_array, label, (x1 + 3, y1+18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw keypoints
        for idx, kp, score in keypoints:
            if kp is not None:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(img_array, (x, y), 3, (255, 255, 255), -1)
                cv2.circle(img_array, (x, y), 5, color, 1)
        
        # Draw skeleton connections
        for connection in SKELETON:
            start_idx, end_idx = connection
            start_kp = None
            end_kp = None
            
            for idx, kp, score in keypoints:
                if idx == start_idx - 1:  # Convert to 0-based indexing
                    start_kp = kp
                elif idx == end_idx - 1:
                    end_kp = kp
            
            if start_kp is not None and end_kp is not None:
                start_point = (int(start_kp[0]), int(start_kp[1]))
                end_point = (int(end_kp[0]), int(end_kp[1]))
                cv2.line(img_array, start_point, end_point, color, 1)
    
    # Convert back to RGB for PIL
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_array)

def run_pose(image_path, detector_model, detector_processor, pose):
    """Main function to process an image"""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Detect humans
    human_boxes = detect_humans(image, detector_model, detector_processor)
    
    if not human_boxes:
        print(f"No humans detected in {image_path}")
        return
    
    all_keypoints = []
    stances = []
    
    # Process each detected human
    for box in human_boxes:
        keypoints = estimate_pose(image, box, pose)
        stance = classify_stance(keypoints)
        
        all_keypoints.append(keypoints)
        stances.append(stance)
        
        print(f"Human detected - Stance: {stance}")
    
    # Visualize results
    result_image = visualize_results(image, human_boxes, all_keypoints, stances)
    
    # Save result
    output_path = str(image_path).replace('.', '_result.')
    result_image.save(output_path)
    print(f"Result saved to {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    args = ap.parse_args()

    # Load models
    detector_model, detector_processor, pose = load_models()
    
    # Process each image
    for image_path in args.images:
        print(f"\nProcessing {image_path}...")
        run_pose(Path(image_path), detector_model, detector_processor, pose)

if __name__ == "__main__":
    main()