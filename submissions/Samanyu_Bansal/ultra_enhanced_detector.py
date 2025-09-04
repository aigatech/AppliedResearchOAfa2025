"""
Ultra-Enhanced Smile Detection with Data Augmentation and Advanced Preprocessing
This version includes data augmentation, advanced preprocessing, and multiple HuggingFace models.
"""

import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image, ImageEnhance, ImageFilter
import warnings
warnings.filterwarnings("ignore")

class UltraEnhancedSmileDetector:
    def __init__(self):
        """Initialize with advanced preprocessing and multiple HuggingFace models."""
        print("🚀 Initializing Ultra-Enhanced Smile Detector...")
        
        self.models = {}
        
        # Load multiple HuggingFace models for ensemble
        self.emotion_models = []
        
        # Model 1: Facial emotion detection
        try:
            print("📥 Loading facial emotion model...")
            model1 = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
            self.emotion_models.append(('facial_emotion', model1))
            print("✅ Facial emotion model loaded")
        except Exception as e:
            print(f"⚠️ Facial emotion model failed: {e}")
        
        # Model 2: ViT face expression
        try:
            print("📥 Loading ViT face expression model...")
            model2 = pipeline("image-classification", model="trpakov/vit-face-expression") 
            self.emotion_models.append(('vit_expression', model2))
            print("✅ ViT expression model loaded")
        except Exception as e:
            print(f"⚠️ ViT expression model failed: {e}")
        
        # Enhanced face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        print(f"🎯 Ultra-enhanced detector ready with {len(self.emotion_models)} specialized models!")
    
    def preprocess_image(self, image):
        """Advanced image preprocessing for better accuracy."""
        # Convert to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        
        # Enhancement techniques
        enhanced_images = []
        
        # Original image
        enhanced_images.append(('original', pil_image))
        
        # Contrast enhancement
        try:
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.2)
            enhanced_images.append(('contrast', enhanced))
        except:
            pass
        
        # Brightness adjustment
        try:
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(1.1)
            enhanced_images.append(('brightness', enhanced))
        except:
            pass
        
        # Sharpness enhancement
        try:
            enhancer = ImageEnhance.Sharpness(pil_image)
            enhanced = enhancer.enhance(1.2)
            enhanced_images.append(('sharpness', enhanced))
        except:
            pass
        
        return enhanced_images
    
    def detect_faces_ultra(self, image):
        """Ultra-robust face detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better detection
        gray_eq = cv2.equalizeHist(gray)
        
        # Multiple detection strategies
        all_faces = []
        
        # Strategy 1: Standard detection on original
        faces1 = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        all_faces.extend(faces1)
        
        # Strategy 2: Detection on equalized image
        faces2 = self.face_cascade.detectMultiScale(gray_eq, 1.1, 4, minSize=(30, 30))
        all_faces.extend(faces2)
        
        # Strategy 3: More sensitive detection
        faces3 = self.face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
        all_faces.extend(faces3)
        
        # Strategy 4: Profile detection
        faces4 = self.profile_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        all_faces.extend(faces4)
        
        # Remove duplicates and return best faces
        if len(all_faces) == 0:
            return []
        
        # Merge overlapping detections
        faces_filtered = []
        for face in all_faces:
            x, y, w, h = face
            is_duplicate = False
            for existing in faces_filtered:
                ex, ey, ew, eh = existing
                # Check for significant overlap
                if (abs(x - ex) < w/2 and abs(y - ey) < h/2):
                    is_duplicate = True
                    break
            if not is_duplicate:
                faces_filtered.append(face)
        
        return faces_filtered
    
    def analyze_with_augmentation(self, face_roi):
        """Analyze face with data augmentation for robustness."""
        # Preprocess image with multiple enhancements
        enhanced_images = self.preprocess_image(face_roi)
        
        all_predictions = []
        confidence_scores = []
        
        # Run each model on each enhanced version
        for model_name, model in self.emotion_models:
            model_predictions = []
            
            for enhancement_name, enhanced_image in enhanced_images:
                try:
                    predictions = model(enhanced_image)
                    if not isinstance(predictions, list):
                        predictions = [predictions]
                    
                    model_predictions.extend(predictions)
                    
                except Exception as e:
                    print(f"⚠️ {model_name} failed on {enhancement_name}: {str(e)[:30]}...")
                    continue
            
            if model_predictions:
                all_predictions.append({
                    'model': model_name,
                    'predictions': model_predictions,
                    'count': len(model_predictions)
                })
        
        return all_predictions
    
    def advanced_ensemble_classification(self, augmented_results):
        """Advanced ensemble classification with weighted voting."""
        emotion_accumulator = {}
        total_weight = 0
        
        # Process all predictions with weights
        for model_result in augmented_results:
            model_name = model_result['model']
            predictions = model_result['predictions']
            
            # Model-specific weights (can be tuned based on model performance)
            model_weight = 1.0
            if 'facial_emotion' in model_name:
                model_weight = 1.2  # Higher weight for specialized facial emotion model
            elif 'vit' in model_name:
                model_weight = 1.1  # Slightly higher for vision transformer
            
            for pred in predictions:
                emotion = pred['label'].lower()
                score = pred['score'] * model_weight
                
                if emotion not in emotion_accumulator:
                    emotion_accumulator[emotion] = 0
                emotion_accumulator[emotion] += score
                total_weight += model_weight
        
        # Normalize scores
        if total_weight > 0:
            for emotion in emotion_accumulator:
                emotion_accumulator[emotion] /= total_weight
        
        # Advanced emotion-to-smile mapping
        smile_scores = {
            'genuine': 0.0,
            'fake': 0.0,
            'no_smile': 0.0
        }
        
        for emotion, score in emotion_accumulator.items():
            # High confidence genuine smile indicators
            if any(word in emotion for word in ['happy', 'joy', 'smile', 'pleased', 'content', 'cheerful']):
                smile_scores['genuine'] += score * 0.9
            
            # Moderate genuine indicators
            elif any(word in emotion for word in ['surprise']):
                smile_scores['genuine'] += score * 0.3
                smile_scores['fake'] += score * 0.4
            
            # Fake smile indicators
            elif any(word in emotion for word in ['fear', 'disgust', 'contempt']):
                smile_scores['fake'] += score * 0.7
            
            # Negative emotions
            elif any(word in emotion for word in ['sad', 'angry', 'mad', 'upset']):
                smile_scores['fake'] += score * 0.5
                smile_scores['no_smile'] += score * 0.4
            
            # Neutral indicators
            elif any(word in emotion for word in ['neutral', 'calm', 'normal']):
                smile_scores['no_smile'] += score * 0.8
        
        # Determine final classification with confidence boosting
        max_score = max(smile_scores.values())
        best_class = max(smile_scores, key=smile_scores.get)
        
        # Confidence calculation with ensemble and augmentation bonuses
        num_models = len(augmented_results)
        num_predictions = sum(r['count'] for r in augmented_results)
        
        ensemble_bonus = min(0.15, num_models * 0.05)
        augmentation_bonus = min(0.10, num_predictions * 0.01)
        
        confidence = min(0.95, max_score + ensemble_bonus + augmentation_bonus)
        
        # Classification with improved thresholds
        if best_class == 'genuine' and confidence > 0.35:
            return "Genuine smile", confidence
        elif best_class == 'fake' and confidence > 0.30:
            return "Fake smile", confidence
        elif best_class == 'no_smile' and confidence > 0.35:
            return "No clear smile", confidence
        else:
            return "Uncertain", confidence
    
    def analyze_image_ultra(self, image_path):
        """Ultra-enhanced analysis with all improvements."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Could not load image"}
        
        # Ultra-robust face detection
        faces = self.detect_faces_ultra(image)
        if len(faces) == 0:
            return {"error": "No faces detected"}
        
        # Use the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract face with optimal padding
        padding = max(20, min(w, h) // 4)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(image.shape[1], x + w + padding)
        y_end = min(image.shape[0], y + h + padding)
        
        face_roi = image[y_start:y_end, x_start:x_end]
        
        # Augmented analysis
        augmented_results = self.analyze_with_augmentation(face_roi)
        
        # Advanced ensemble classification
        classification, confidence = self.advanced_ensemble_classification(augmented_results)
        
        # Calculate total predictions used
        total_predictions = sum(r['count'] for r in augmented_results)
        
        return {
            "classification": classification,
            "confidence": confidence,
            "face_coords": largest_face,
            "models_used": len(self.emotion_models),
            "total_predictions": total_predictions,
            "augmented_results": augmented_results,
            "method": "Ultra-Enhanced HuggingFace Ensemble with Data Augmentation"
        }

def test_ultra_detector():
    """Test the ultra-enhanced detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Enhanced Smile Detection')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image')
    parser.add_argument('--show_details', action='store_true', help='Show detailed analysis')
    
    args = parser.parse_args()
    
    detector = UltraEnhancedSmileDetector()
    results = detector.analyze_image_ultra(args.image_path)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    print(f"\n🎯 ULTRA-ENHANCED SMILE ANALYSIS")
    print(f"=" * 60)
    print(f"Classification: {results['classification']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Models Used: {results['models_used']} HuggingFace models")
    print(f"Total Predictions: {results['total_predictions']} (with augmentation)")
    print(f"Method: {results['method']}")
    
    if args.show_details and 'augmented_results' in results:
        print(f"\n📊 DETAILED ANALYSIS:")
        for model_result in results['augmented_results']:
            model_name = model_result['model']
            count = model_result['count']
            print(f"\n🤖 {model_name.upper()} ({count} predictions):")
            
            # Show top emotions from this model
            emotions = {}
            for pred in model_result['predictions']:
                emotion = pred['label'].lower()
                score = pred['score']
                if emotion not in emotions:
                    emotions[emotion] = []
                emotions[emotion].append(score)
            
            # Average and display top emotions
            avg_emotions = {e: np.mean(scores) for e, scores in emotions.items()}
            top_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for emotion, avg_score in top_emotions:
                print(f"   {emotion}: {avg_score:.3f}")

if __name__ == "__main__":
    test_ultra_detector()
