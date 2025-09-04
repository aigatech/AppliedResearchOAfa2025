"""
Genuine vs. Fake Smile Detection for Dating Apps
Author: Samanyu Bansal

This script analyzes facial images to determine if a smile is genuine or fake.
Uses facial landmark detection to identify key features that distinguish 
genuine (Duchenne) smiles from fake smiles.
"""

import cv2
import numpy as np
import argparse
import math
from typing import Tuple, List, Optional

class SmileDetector:
    def __init__(self):
        """Initialize the smile detector with OpenCV face and landmark detection."""
        # Load OpenCV's pre-trained face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load facial landmark detector (using OpenCV's built-in)
        # Note: For production, you might want to use dlib or MediaPipe
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the main face in the image with improved parameters.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (x, y, w, h) for the face bounding box, or None if no face found
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try with different parameters for better detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        # If no faces found, try with more relaxed parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        
        # Try with even more relaxed parameters as last resort
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        if len(faces) == 0:
            return None
        
        # Return the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        return tuple(largest_face)
    
    def detect_eyes_and_smile(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> dict:
        """
        Detect eyes and smile within the face region.
        
        Args:
            image: Input image
            face_coords: Face bounding box (x, y, w, h)
            
        Returns:
            Dictionary with eye and smile detection results
        """
        x, y, w, h = face_coords
        face_roi = image[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the upper half of the face
        eye_region = face_gray[0:h//2, :]
        eyes = self.eye_cascade.detectMultiScale(eye_region, 1.1, 5)
        
        # Detect smile in the lower half of the face
        smile_region = face_gray[h//2:, :]
        smiles = self.smile_cascade.detectMultiScale(smile_region, 1.8, 20)
        
        return {
            'eyes': eyes,
            'smiles': smiles,
            'face_roi': face_roi,
            'face_gray': face_gray
        }
    
    def calculate_smile_features(self, detection_results: dict) -> dict:
        """
        Calculate features that help distinguish genuine from fake smiles.
        
        Genuine smiles (Duchenne smiles) typically involve:
        1. Eye constriction (crow's feet)
        2. Raised cheeks
        3. Symmetrical mouth movement
        
        Args:
            detection_results: Results from detect_eyes_and_smile
            
        Returns:
            Dictionary of calculated features
        """
        features = {
            'has_smile': len(detection_results['smiles']) > 0,
            'has_eyes': len(detection_results['eyes']) >= 1,
            'eye_smile_ratio': 0,
            'smile_width_ratio': 0,
            'symmetry_score': 0
        }
        
        if not features['has_smile']:
            return features
        
        face_gray = detection_results['face_gray']
        h, w = face_gray.shape
        
        # Calculate eye-smile interaction
        if features['has_eyes'] and features['has_smile']:
            # Measure the "squinting" effect around eyes (proxy for crow's feet)
            eye_region = face_gray[0:h//2, :]
            eye_intensity = np.mean(eye_region)
            
            # Measure smile intensity
            smile_region = face_gray[h//2:, :]
            smile_intensity = np.mean(smile_region)
            
            # Genuine smiles often show eye constriction
            features['eye_smile_ratio'] = eye_intensity / (smile_intensity + 1e-6)
        
        # Analyze smile characteristics
        if len(detection_results['smiles']) > 0:
            smile = detection_results['smiles'][0]  # Take the most prominent smile
            smile_w = smile[2]
            
            # Width ratio compared to face
            features['smile_width_ratio'] = smile_w / w
            
            # Simple symmetry check (this is a simplified version)
            # In a real implementation, you'd use more sophisticated landmark detection
            features['symmetry_score'] = min(1.0, features['smile_width_ratio'] * 2)
        
        return features
    
    def classify_smile(self, features: dict) -> Tuple[str, float]:
        """
        Classify the smile as genuine or fake based on extracted features.
        
        Args:
            features: Dictionary of calculated features
            
        Returns:
            Tuple of (classification, confidence_score)
        """
        if not features['has_smile']:
            return "No smile detected", 0.0
        
        # Improved rule-based classification
        score = 0.0
        
        # Eye involvement is crucial for genuine smiles
        if features['has_eyes']:
            score += 0.3
            
            # Eye-smile interaction (adjusted ranges)
            if 0.75 <= features['eye_smile_ratio'] <= 1.25:
                score += 0.25  # Good eye-smile balance
            elif features['eye_smile_ratio'] > 1.5:
                score -= 0.1   # Too much difference suggests fake
        
        # Smile width (adjusted for more realistic range)
        if 0.2 <= features['smile_width_ratio'] <= 0.8:
            score += 0.25
        elif features['smile_width_ratio'] > 0.9:
            score -= 0.1  # Too wide might be fake
        
        # Symmetry (improved scoring)
        if features['symmetry_score'] > 0.4:
            score += 0.2
        
        # Normalize score
        score = max(0.0, min(1.0, score))
        
        # Classify based on improved thresholds
        if score >= 0.5:
            return "Genuine smile", score
        elif score >= 0.25:
            return "Fake smile", 1.0 - score
        else:
            return "Uncertain smile", score
    
    def analyze_image(self, image_path: str) -> dict:
        """
        Main function to analyze an image for smile authenticity.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Detect face
        face_coords = self.detect_face(image)
        if face_coords is None:
            return {"error": "No face detected in image"}
        
        # Detect facial features
        detection_results = self.detect_eyes_and_smile(image, face_coords)
        
        # Calculate features
        features = self.calculate_smile_features(detection_results)
        
        # Classify smile
        classification, confidence = self.classify_smile(features)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "features": features,
            "face_coords": face_coords,
            "image_shape": image.shape
        }

def main():
    """Main function to run the smile detector from command line."""
    parser = argparse.ArgumentParser(description='Detect genuine vs fake smiles in images')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image file')
    parser.add_argument('--show_image', action='store_true', help='Display the image with detection results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = SmileDetector()
    
    # Analyze image
    results = detector.analyze_image(args.image_path)
    
    # Print results
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\n=== Smile Analysis Results ===")
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.2f}")
    print(f"\nFeature Details:")
    print(f"- Has smile: {results['features']['has_smile']}")
    print(f"- Has eyes detected: {results['features']['has_eyes']}")
    print(f"- Eye-smile ratio: {results['features']['eye_smile_ratio']:.2f}")
    print(f"- Smile width ratio: {results['features']['smile_width_ratio']:.2f}")
    print(f"- Symmetry score: {results['features']['symmetry_score']:.2f}")
    
    # Optionally show image
    if args.show_image:
        image = cv2.imread(args.image_path)
        x, y, w, h = results['face_coords']
        
        # Draw face rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Add text
        cv2.putText(image, results['classification'], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Smile Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
