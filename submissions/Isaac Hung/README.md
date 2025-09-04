# Road Perception

**Note: this README.md is also in the Jupyter Notebook [RoadPerception.ipynb](/submissions/Isaac%20Hung/RoadPerception.ipynb).**

Perception is an integral component of any form of robot or autonomous vehicle system, including the systems that may be used in an embodied agent. This project demonstrates some perception techniques that could be used in a road setting.

## What is it?

Given an image showing the front camera view from a car on a road, this project detects the road lane markings and other cars, then identifies the next car in the current lane. This information has several applications; for example, it could be used to tell the robot whether it is too close, too far or at an appropriate distance from the vehicle in front. Alternatively, the lane markings could also be used to follow a given trajectory or determine whether or not the car is drifting out of lane.

## How does it work?

Two models from Hugging Face are used: [TwinLiteNetPlus](https://huggingface.co/nielsr/twinlitenetplus-nano) for lane detection, and [RT-DETRv2](https://huggingface.co/PekingU/rtdetr_v2_r101vd) for object detection. TwinLiteNetPlus is a very lightweight model that can produce binary segmentation masks for lane markers, which are further preprocessed in this project. RT-DETRv2 is a robust transformer-based object detection model, which has the advantage of not requiring non-maximum suppression (NMS), unlike the YOLO family of models. Both of these models feature real-time performance characteristics, making them suitable for robots and an embodied agent.

The lane detection pipeline works first by finding contours in the segmentation map to find the centroids of the keypoints on the lanes. Then, cubic splines are fitted to these points both to remove noise by smoothing out the results and also to interpolate gaps in between keypoints. To identify individual lane markings, the angle between the line drawn from a keypoint to the vanishing point (which all lane lines converge towards) and the horizontal is taken, since this identifies which lane marking the keypoint belongs to. Then, a simple algorithm is used to "cluster" the points before interpolation.

To identify the next car in the lane, RT-DETRv2 is used to detect car objects, returning their bounding boxes. The bottom-center point on the bounding box is taken. Using the left and right lane marker identified through lane detection, the objects are filtered for whether or not they are in the same lane as the ego vehicle, and finally the nearest car is taken.

## How to run it?

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the code in the Jupyter Notebook [RoadPerception.ipynb](/submissions/Isaac%20Hung/RoadPerception.ipynb).

Notes:
- This code was developed and tested using Python 3.13.
- It is also possible to use the uv package manager to manage dependencies.