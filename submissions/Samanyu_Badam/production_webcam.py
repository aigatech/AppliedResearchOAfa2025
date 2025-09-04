"""
Production Webcam Smile Detection - Real HuggingFace Model Analysis
No demo mode - pure production-ready real-time smile detection
"""

import cv2
import os
import time
import numpy as np
from enhanced_detector import EnhancedSmileDetector

class ProductionWebcamAnalyzer:
    def __init__(self):
        """Initialize production webcam analyzer with HuggingFace models."""
        print("🚀 Initializing Production Webcam Smile Analyzer...")
        print("Loading HuggingFace models for real-time analysis...")
        
        # Initialize the enhanced detector with multiple HuggingFace models
        self.detector = EnhancedSmileDetector()
        
        print("✅ Production analyzer ready!")
    
    def start_real_time_analysis(self):
        """Start real-time webcam analysis with HuggingFace models."""
        print("\n📹 Starting Real-Time Webcam Analysis")
        print("=" * 50)
        print("• Using HuggingFace models for smile detection")
        print("• Press 'q' to quit")
        print("• Press 's' to save current analysis")
        print("• Analysis runs continuously every 1 second")
        print("=" * 50)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not access webcam")
            return
        
        # Analysis settings
        last_analysis_time = 0
        analysis_interval = 1.0  # Analyze every 1 second
        frame_count = 0
        
        # Store results for display
        current_result = {
            'classification': 'Initializing...',
            'confidence': 0.0,
            'models_used': 0
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = time.time()
            display_frame = frame.copy()
            
            # Perform real HuggingFace model analysis
            if (current_time - last_analysis_time) > analysis_interval:
                print(f"\n🔍 Frame {frame_count}: Analyzing with HuggingFace models...")
                
                # Save frame for analysis
                temp_path = "webcam_analysis_temp.jpg"
                cv2.imwrite(temp_path, frame)
                
                # Real HuggingFace model analysis
                try:
                    results = self.detector.analyze_image_enhanced(temp_path)
                    
                    if "error" not in results:
                        current_result = {
                            'classification': results['classification'],
                            'confidence': results['confidence'],
                            'models_used': results['models_used']
                        }
                        
                        # Print real analysis results
                        print(f"✅ Classification: {results['classification']}")
                        print(f"   Confidence: {results['confidence']:.3f}")
                        print(f"   Models Used: {results['models_used']} HuggingFace models")
                        
                        # Draw face detection box
                        if 'face_coords' in results:
                            x, y, w, h = results['face_coords']
                            
                            # Color based on classification
                            if 'Genuine' in results['classification']:
                                color = (0, 255, 0)  # Green
                            elif 'Fake' in results['classification']:
                                color = (0, 165, 255)  # Orange
                            else:
                                color = (255, 255, 255)  # White
                            
                            # Draw face rectangle
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            
                            # Add classification text
                            cv2.putText(display_frame, f"{results['classification']}", 
                                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(display_frame, f"Confidence: {results['confidence']:.2f}", 
                                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        current_result['classification'] = results['error']
                        print(f"⚠️ {results['error']}")
                        
                        # Display error on frame
                        cv2.putText(display_frame, results['error'], 
                                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                except Exception as e:
                    error_msg = f"Analysis Error: {str(e)[:30]}..."
                    current_result['classification'] = error_msg
                    print(f"❌ {error_msg}")
                    
                    cv2.putText(display_frame, "Model Error", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                last_analysis_time = current_time
                frame_count += 1
            
            # Always display current status
            status_text = f"HuggingFace Analysis: {current_result['classification']}"
            cv2.putText(display_frame, status_text, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            models_text = f"Models: {current_result['models_used']} HuggingFace"
            cv2.putText(display_frame, models_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Production Webcam Smile Detection - HuggingFace Models', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current analysis frame
                timestamp = int(time.time())
                save_path = f"smile_analysis_{timestamp}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"💾 Saved analysis: {save_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Real-time analysis completed.")

def main():
    """Main function to start production webcam analysis."""
    try:
        analyzer = ProductionWebcamAnalyzer()
        analyzer.start_real_time_analysis()
    except KeyboardInterrupt:
        print("\n🛑 Analysis stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
