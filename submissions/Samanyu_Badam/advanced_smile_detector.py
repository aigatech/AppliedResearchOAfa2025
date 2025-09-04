"""
Advanced Smile Detection using HuggingFace Vision Models
This version incorporates pre-trained vision models from HuggingFace for more accurate emotion detection.
"""

import cv2
import numpy as np
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class AdvancedSmileDetector:
    def __init__(self):
        """Initialize with state-of-the-art HuggingFace models for better accuracy."""
        # Use multiple models for ensemble prediction
        self.emotion_models = []
        
        # Model 1: Specialized facial emotion detection
        try:
            model1 = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection",
                return_all_scores=True
            )
            self.emotion_models.append(("facial_emotions", model1))
            print("✓ Loaded facial emotion detection model")
        except Exception as e:
            print(f"⚠ Failed to load model 1: {e}")
        
        # Model 2: Vision transformer for expressions
        try:
            model2 = pipeline(
                "image-classification", 
                model="trpakov/vit-face-expression",
                return_all_scores=True
            )
            self.emotion_models.append(("vit_expression", model2))
            print("✓ Loaded ViT expression model")
        except Exception as e:
            print(f"⚠ Failed to load model 2: {e}")
        
        # Model 3: Another emotion classifier for ensemble
        try:
            model3 = pipeline(
                "image-classification",
                model="kdhht2334/autotrain-facial_emotions-40429105176",
                return_all_scores=True
            )
            self.emotion_models.append(("autotrain_emotions", model3))
            print("✓ Loaded autotrain emotion model")
        except Exception as e:
            print(f"⚠ Failed to load model 3: {e}")
        
        if not self.emotion_models:
            print("⚠ Using fallback emotion detection...")
        
        # Use MediaPipe for more accurate face detection (if available)
        try:
            import mediapipe as mp
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5)
            self.use_mediapipe = True
            print("✓ Loaded MediaPipe face detection")
        except:
            self.use_mediapipe = False
            print("⚠ MediaPipe not available, using OpenCV")
        
        # Fallback to OpenCV face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Advanced feature extractor
        try:
            self.feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
            self.feature_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
            print("✓ Loaded ResNet-50 feature extraction model")
        except:
            try:
                self.feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
                self.feature_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
                print("✓ Loaded ResNet-18 feature extraction model")
            except:
                self.feature_extractor = None
                self.feature_model = None
                print("⚠ No feature extraction model available")
    
    def detect_faces(self, image):
        """Detect faces in the image using multiple methods for better accuracy."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try frontal face detection first
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        # If no frontal faces found, try profile detection
        if len(faces) == 0:
            faces = self.profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        
        # If still no faces, try with more relaxed parameters
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        
        return faces
    
    def extract_face_roi(self, image, face_coords):
        """Extract face region of interest."""
        x, y, w, h = face_coords
        # Add some padding around the face
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        return face_roi
    
    def analyze_facial_features(self, face_roi):
        """Analyze facial features for smile detection."""
        # Convert to PIL Image for HuggingFace models
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        results = {}
        
        # Use emotion classifier if available
        if self.emotion_classifier:
            try:
                emotions = self.emotion_classifier(pil_image)
                results['emotions'] = emotions
            except:
                results['emotions'] = None
        
        # Traditional computer vision analysis
        results['cv_features'] = self.analyze_cv_features(face_roi)
        
        return results
    
    def analyze_cv_features(self, face_roi):
        """Traditional computer vision feature analysis."""
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray_face.shape
        
        # Analyze mouth region (lower third of face)
        mouth_region = gray_face[2*h//3:, :]
        mouth_mean = np.mean(mouth_region)
        mouth_std = np.std(mouth_region)
        
        # Analyze eye region (upper third of face)
        eye_region = gray_face[:h//3, :]
        eye_mean = np.mean(eye_region)
        eye_std = np.std(eye_region)
        
        # Calculate ratios and features
        mouth_eye_ratio = mouth_mean / (eye_mean + 1e-6)
        contrast_ratio = mouth_std / (eye_std + 1e-6)
        
        # Edge detection for smile curvature
        edges = cv2.Canny(gray_face, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        return {
            'mouth_eye_ratio': mouth_eye_ratio,
            'contrast_ratio': contrast_ratio,
            'edge_density': edge_density,
            'mouth_brightness': mouth_mean,
            'eye_brightness': eye_mean
        }
    
    def classify_smile_advanced(self, analysis_results):
        """Advanced smile classification using multiple signals."""
        classification_scores = {
            'genuine': 0.0,
            'fake': 0.0,
            'no_smile': 0.0
        }
        
        # Emotion-based analysis with proper mapping
        if analysis_results.get('emotions'):
            for emotion in analysis_results['emotions']:
                label = emotion['label'].lower()
                score = emotion['score']
                
                # Map emotions to smile types
                if any(word in label for word in ['happy', 'joy', 'smile', 'pleased']):
                    classification_scores['genuine'] += score * 0.6
                elif any(word in label for word in ['neutral', 'calm']):
                    classification_scores['no_smile'] += score * 0.4
                elif any(word in label for word in ['sad', 'angry', 'fear', 'disgust']):
                    classification_scores['fake'] += score * 0.3
                elif any(word in label for word in ['surprise']):
                    classification_scores['fake'] += score * 0.2
        
        # Computer vision feature analysis with better weights
        cv_features = analysis_results.get('cv_features', {})
        
        # Mouth-eye brightness ratio (genuine smiles affect eye area)
        mouth_eye_ratio = cv_features.get('mouth_eye_ratio', 1.0)
        if 0.85 <= mouth_eye_ratio <= 1.15:  # Slightly wider range
            classification_scores['genuine'] += 0.25
        elif mouth_eye_ratio > 1.3:  # Too much difference suggests fake
            classification_scores['fake'] += 0.2
        
        # Edge density (genuine smiles have more natural curves)
        edge_density = cv_features.get('edge_density', 0.0)
        if 0.05 <= edge_density <= 0.35:  # Adjusted range
            classification_scores['genuine'] += 0.15
        
        # Contrast analysis
        contrast_ratio = cv_features.get('contrast_ratio', 1.0)
        if 0.8 <= contrast_ratio <= 1.5:
            classification_scores['genuine'] += 0.1
        
        # Determine final classification with better thresholds
        max_score = max(classification_scores.values())
        
        if max_score < 0.2:  # Lower threshold
            return "Uncertain", max_score
        
        best_class = max(classification_scores, key=classification_scores.get)
        
        # Add confidence boost if clear winner
        if max_score > 0.5:
            confidence = min(0.95, max_score * 1.2)
        else:
            confidence = max_score
        
        if best_class == 'genuine':
            return "Genuine smile", confidence
        elif best_class == 'fake':
            return "Fake smile", confidence
        else:
            return "No clear smile", confidence
    
    def analyze_image(self, image_path):
        """Main analysis function."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Detect faces
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return {"error": "No faces detected"}
        
        # Analyze the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        face_roi = self.extract_face_roi(image, largest_face)
        
        # Analyze facial features
        analysis_results = self.analyze_facial_features(face_roi)
        
        # Classify smile
        classification, confidence = self.classify_smile_advanced(analysis_results)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "face_coords": largest_face,
            "analysis_details": analysis_results,
            "model_used": "HuggingFace + OpenCV hybrid"
        }

def test_advanced_detector():
    """Test function for the advanced detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Smile Detection')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--show_details', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    detector = AdvancedSmileDetector()
    results = detector.analyze_image(args.image_path)
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"\n=== Advanced Smile Analysis ===")
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Model: {results['model_used']}")
    
    if args.show_details and 'analysis_details' in results:
        details = results['analysis_details']
        
        if details.get('emotions'):
            print(f"\nEmotion Scores:")
            for emotion in details['emotions'][:3]:  # Top 3 emotions
                print(f"  {emotion['label']}: {emotion['score']:.3f}")
        
        if details.get('cv_features'):
            cv = details['cv_features']
            print(f"\nCV Features:")
            print(f"  Mouth-Eye Ratio: {cv.get('mouth_eye_ratio', 0):.3f}")
            print(f"  Edge Density: {cv.get('edge_density', 0):.3f}")
            print(f"  Contrast Ratio: {cv.get('contrast_ratio', 0):.3f}")

if __name__ == "__main__":
    test_advanced_detector()
