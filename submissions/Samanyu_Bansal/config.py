"""
Configuration and utilities for the Smile Detection project
"""

# Model configurations
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
VISION_MODEL = "microsoft/resnet-18"

# Classification thresholds
CONFIDENCE_THRESHOLD = 0.3
GENUINE_SMILE_THRESHOLD = 0.6

# Feature analysis parameters
MOUTH_EYE_RATIO_RANGE = (0.9, 1.1)
EDGE_DENSITY_RANGE = (0.1, 0.3)

# Face detection parameters
FACE_CASCADE_SCALE_FACTOR = 1.1
FACE_CASCADE_MIN_NEIGHBORS = 4

# Display settings
WINDOW_SIZE = (800, 600)
FONT_SCALE = 0.7
FONT_COLOR = (255, 0, 0)
BBOX_COLOR = (255, 0, 0)
BBOX_THICKNESS = 2

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Error messages
ERROR_MESSAGES = {
    'no_image': "Could not load image. Please check the file path.",
    'no_face': "No faces detected in the image. Please ensure the image contains a clear face.",
    'model_error': "Error loading ML models. Please check your internet connection.",
    'processing_error': "Error processing the image. Please try a different image."
}

# Success messages
SUCCESS_MESSAGES = {
    'genuine': "😊 Genuine smile detected! This appears to be an authentic expression.",
    'fake': "🤔 Fake smile detected. The smile appears forced or artificial.",
    'uncertain': "😐 Uncertain classification. The smile is ambiguous."
}
