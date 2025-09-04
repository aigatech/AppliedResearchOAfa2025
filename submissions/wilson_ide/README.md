# Human Pose Estimation and Stance Classification

This project uses YolosForObjectDetection to detect human bodies in images, MediaPipe for human pose estimation, and applies heuristics to classify human stances as either "active" or "passive" based on keypoint positions and body language.

## Features

- **Human Detection**: Uses YOLOS to detect humans in images
- **Pose Estimation**: Uses MediaPipe Pose to extract keypoints of a person
- **Stance Classification**: Applies 4 main heuristics to classify stance as aggressive or passive based on the relation of body parts
- **Visualization**: Draws keypoints, skeleton connections, and stance labels on output images

## Stance Classification Heuristics

The system classifies stance based on several heuristics:

### 1. Arm Positions


### 2. Stance Width


### 3. Posture


### 4. Leg activity


## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Import an image to the images folder and run the script:

```bash
python main.py --images images/image2.png
```

## Output

For each input image, the system will detect the humans, find their main joints, use that to classify them each, and output an image showing the skeleton and whether they are active or passive. The terminal will spit out whether they are active or passive as well as their activity score, and the image will outline them along with text saying which they are.



## Model Details

- **Object Detection**: `hustvl/yolos-tiny` - Detects humans in images
- **Pose Estimation**: MediaPipe Pose - Extracts 17 keypoints per person
