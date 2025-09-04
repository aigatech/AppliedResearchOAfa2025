"""
Enhanced Smile Detection using Multiple HuggingFace Models
This version uses ensemble learning with multiple HuggingFace models for improved accuracy.
"""

import cv2
import numpy as np
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class EnhancedSmileDetector:
    def __init__(self):
        """Initialize with multiple HuggingFace models for ensemble prediction."""
        print("🚀 Initializing Enhanced Smile Detector with HuggingFace models...")
        
        self.models = {}
        self.pipelines = {}
        
        # 1. Primary facial emotion detection model
        try:
            print("📥 Loading primary emotion model...")
            self.pipelines['emotion1'] = pipeline(
                "image-classification",
                model="dima806/facial_emotions_image_detection"
            )
            print("✅ Primary emotion model loaded")
        except Exception as e:
            print(f"⚠️ Primary model failed: {e}")
            self.pipelines['emotion1'] = None
        
        # 2. Alternative emotion detection model
        try:
            print("📥 Loading alternative emotion model...")
            self.pipelines['emotion2'] = pipeline(
                "image-classification", 
                model="trpakov/vit-face-expression"
            )
            print("✅ Alternative emotion model loaded")
        except Exception as e:
            print(f"⚠️ Alternative model failed: {e}")
            self.pipelines['emotion2'] = None
        
        # 3. Vision Transformer for features
        try:
            print("📥 Loading ViT feature extractor...")
            self.pipelines['vit'] = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224"
            )
            print("✅ ViT model loaded")
        except Exception as e:
            print(f"⚠️ ViT model failed: {e}")
            self.pipelines['vit'] = None
        
        # Face detection with multiple cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        active_models = sum(1 for p in self.pipelines.values() if p is not None)
        print(f"🎯 Enhanced detector ready with {active_models} HuggingFace models!")
    
    def detect_faces_enhanced(self, image):
        """Enhanced face detection with multiple methods."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection strategies
        faces = []
        
        # Strategy 1: Standard frontal detection
        faces1 = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        faces.extend(faces1)
        
        # Strategy 2: More sensitive detection
        if len(faces) == 0:
            faces2 = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
            faces.extend(faces2)
        
        # Strategy 3: Profile detection
        if len(faces) == 0:
            faces3 = self.profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            faces.extend(faces3)
        
        # Strategy 4: Very relaxed detection
        if len(faces) == 0:
            faces4 = self.face_cascade.detectMultiScale(gray, 1.3, 2, minSize=(15, 15))
            faces.extend(faces4)
        
        return faces
    
    def analyze_with_ensemble(self, face_image):
        """Analyze image using ensemble of HuggingFace models."""
        results = {
            'emotion_scores': {},
            'predictions': [],
            'confidence_scores': []
        }
        
        # Convert to PIL for HuggingFace models
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        
        # Run predictions with all available models
        for model_name, pipeline in self.pipelines.items():
            if pipeline is None:
                continue
            
            try:
                if model_name == 'emotion3':
                    # Skip text-based models for image analysis
                    continue
                
                # Get top predictions (default behavior)
                predictions = pipeline(pil_image)
                
                # Convert single prediction to list if needed
                if not isinstance(predictions, list):
                    predictions = [predictions]
                
                results['predictions'].append({
                    'model': model_name,
                    'results': predictions
                })
                
                # Extract emotion scores
                for pred in predictions:
                    emotion = pred['label'].lower()
                    score = pred['score']
                    
                    if emotion not in results['emotion_scores']:
                        results['emotion_scores'][emotion] = []
                    results['emotion_scores'][emotion].append(score)
                
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {str(e)[:50]}...")
                continue
        
        return results
    
    def ensemble_classification(self, ensemble_results):
        """Combine results from multiple models for final classification."""
        emotion_scores = ensemble_results['emotion_scores']
        
        # Calculate averaged emotion scores
        avg_emotions = {}
        for emotion, scores in emotion_scores.items():
            avg_emotions[emotion] = np.mean(scores)
        
        # Map emotions to smile types with confidence
        smile_scores = {
            'genuine': 0.0,
            'fake': 0.0,
            'no_smile': 0.0
        }
        
        # Enhanced emotion mapping
        for emotion, score in avg_emotions.items():
            # Genuine smile indicators
            if any(word in emotion for word in ['happy', 'joy', 'smile', 'pleasant', 'content']):
                smile_scores['genuine'] += score * 0.8
            
            # Fake smile indicators  
            elif any(word in emotion for word in ['surprise', 'fear', 'disgusted']):
                smile_scores['fake'] += score * 0.6
            
            # Negative emotions (likely fake or no smile)
            elif any(word in emotion for word in ['sad', 'angry', 'mad']):
                smile_scores['fake'] += score * 0.4
                smile_scores['no_smile'] += score * 0.3
            
            # Neutral states
            elif any(word in emotion for word in ['neutral', 'calm', 'normal']):
                smile_scores['no_smile'] += score * 0.5
        
        # Determine best classification
        max_score = max(smile_scores.values())
        best_class = max(smile_scores, key=smile_scores.get)
        
        # Calculate confidence with ensemble bonus
        num_models = len(ensemble_results['predictions'])
        ensemble_bonus = min(0.2, num_models * 0.05)  # Bonus for using multiple models
        confidence = min(0.95, max_score + ensemble_bonus)
        
        # Final classification
        if best_class == 'genuine' and confidence > 0.4:
            return "Genuine smile", confidence
        elif best_class == 'fake' and confidence > 0.3:
            return "Fake smile", confidence
        elif best_class == 'no_smile' and confidence > 0.3:
            return "No clear smile", confidence
        else:
            return "Uncertain", confidence
    
    def analyze_image_enhanced(self, image_path):
        """Enhanced analysis using ensemble of HuggingFace models."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Detect faces with enhanced method
        faces = self.detect_faces_enhanced(image)
        if len(faces) == 0:
            return {"error": "No faces detected"}
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face region with padding
        padding = 20
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Ensemble analysis
        ensemble_results = self.analyze_with_ensemble(face_roi)
        
        # Final classification
        classification, confidence = self.ensemble_classification(ensemble_results)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "face_coords": largest_face,
            "ensemble_results": ensemble_results,
            "models_used": len(ensemble_results['predictions']),
            "model_type": "HuggingFace Ensemble"
        }

def test_enhanced_detector():
    """Test the enhanced detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced HuggingFace Smile Detection')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--show_details', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    detector = EnhancedSmileDetector()
    results = detector.analyze_image_enhanced(args.image_path)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n🎯 ENHANCED SMILE ANALYSIS")
    print(f"=" * 50)
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Models Used: {results['models_used']} HuggingFace models")
    print(f"Method: {results['model_type']}")
    
    if args.show_details and 'ensemble_results' in results:
        ensemble = results['ensemble_results']
        
        print(f"\n📊 ENSEMBLE DETAILS:")
        for prediction in ensemble['predictions']:
            model_name = prediction['model']
            print(f"\n🤖 {model_name.upper()}:")
            for result in prediction['results'][:3]:  # Top 3 results
                print(f"   {result['label']}: {result['score']:.3f}")
        
        print(f"\n📈 AVERAGED EMOTIONS:")
        for emotion, scores in ensemble['emotion_scores'].items():
            avg_score = np.mean(scores)
            print(f"   {emotion}: {avg_score:.3f} (from {len(scores)} models)")

if __name__ == "__main__":
    test_enhanced_detector()
