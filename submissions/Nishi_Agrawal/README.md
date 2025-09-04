# GT Robot's Eye: A Rivalry-Powered Vision System

Ever wondered what would happen if you gave a robot Georgia Tech spirit? Well, now you know! This fun project combines computer vision with good old-fashioned college rivalry to create a robot that's definitely not a fan of red and black.

Watch it in action here: https://app.screencastify.com/watch/kyBZ7s3NT2neTX7W9xRu

## Key Features

- **Smart Vision**: The robot can spot and identify objects in real-time (DETR)
- **Path Planning**: Draws smooth, curved paths to objects
- **GT Spirit Detection**: 
  - Notifies when it sees GT Gold (enthusiastic "GO JACKETS!" moments)
  - Raises the alarm when it spots that dreaded UGA red
- **Fun Interface**:
  - Watch the robot's view in real-time
  - See it plot paths
  - Get live updates on its GT spirit level
  - Enjoy dramatic reactions to rival colors

## Getting Started

First, copy this into your terminal:
```bash
pip install opencv-python torch transformers pillow numpy tk
```

## Running Your GT-Spirited Robot

1. Ensure packages are installed.
2. Run:
```bash
python app.py
```
3. Press "Start Camera 🎥" button!

## What to Expect

## Color Detection System

**Georgia Tech Color Detection**
- Implements smart detection for iconic Yellow and Gold
- Triggers victory messages when GT colors are spotted
- Enhances detected objects with distinctive highlighting
- Keeps user informed with enthusiatic status updates

**Rival Detection System**
- Maintains vigilent monitoring for unwanted red patterns
- Deploys rapid alert system upon detecttion
- Provides clear visual warning indicators
- Ensures proper notification of potential threats

## Technical Architecture

This system harnesses powerful technologies:
- DETR model integration for precise object detection
- Advanced HSV color space analysis
- Multi-threaded design for exceptional responsiveness
- Sophisticated trajectory planning algorithms

## Core Components

Main application (`app.py`) structure:
- RobotVisionUI class driving the interactive experience
- Real-time detection and analysis pipeline
- Dynamic trajectory computation engine
- Comprehensive video processing system

PSA: No UGAs harmed in the making of this 1.5 hour project!