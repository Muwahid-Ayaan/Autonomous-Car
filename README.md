# Autonomous Lane Following in CARLA using OpenCV and Pure Pursuit

## Overview

This project implements a complete vision-based lane-following system in the CARLA simulator using classical computer vision techniques and geometric vehicle control.

The vehicle navigates autonomously by detecting lane boundaries from a front-facing RGB camera, estimating the lane center, and generating steering commands using the Pure Pursuit path-tracking algorithm.

Unlike modern deep-learning approaches, this project relies entirely on traditional image processing and geometry, making every stage of the pipeline interpretable and computationally lightweight.

---

## Features

* Real-time lane detection from a monocular RGB camera
* Perspective (Bird's Eye View) transformation
* HSV-based lane segmentation
* Histogram peak detection
* Sliding-window lane tracking
* Lane center estimation
* Pure Pursuit steering controller
* Integration with CARLA Simulator
* Real-time visualization and debugging tools

---

## System Architecture

Camera Image
↓
Perspective Transform (BEV)
↓
HSV Thresholding
↓
Binary Lane Mask
↓
Histogram Analysis
↓
Sliding Window Search
↓
Lane Center Estimation
↓
Pure Pursuit Controller
↓
Vehicle Steering Command

---

## Lane Detection Pipeline

### 1. Image Acquisition

Images are captured from a front-mounted RGB camera attached to the ego vehicle in CARLA.

### 2. Perspective Transformation

A bird's-eye-view transformation is applied to remove perspective distortion and simplify lane geometry.

This allows lane markings to appear approximately parallel in the transformed image.

### 3. Lane Segmentation

The transformed image is converted to HSV color space.

A configurable HSV threshold isolates lane markings and generates a binary mask.

### 4. Histogram Analysis

A horizontal histogram is computed over the lower half of the binary mask.

The dominant peaks correspond to the left and right lane boundaries and provide initial search locations.

### 5. Sliding Window Tracking

A multi-window search tracks lane markings vertically through the image.

The centroid of detected lane pixels is used to update window positions and recover lane geometry.

### 6. Lane Center Estimation

The center of the lane is estimated from the detected lane boundaries.

The lateral offset between the vehicle and lane center is then computed.

---

## Vehicle Control

The project uses the Pure Pursuit algorithm for steering control.

A look-ahead point is generated relative to the vehicle:

* Forward distance: 5 meters
* Lateral offset: computed from lane center estimation

The Pure Pursuit controller converts this target point into a curvature command, which is then translated into steering input for the vehicle.

---

## Technologies Used

### Simulation

* CARLA Simulator

### Computer Vision

* OpenCV
* NumPy

### Control

* Pure Pursuit Path Tracking

### Programming Language

* Python

---

## Performance

The system is capable of:

* Real-time lane detection
* Real-time steering control
* Autonomous lane following on standard CARLA roads

The complete pipeline operates using classical computer vision methods without requiring GPU-accelerated neural networks.

---

## Limitations

As a traditional vision-based approach, performance may degrade under:

* Extreme lighting changes
* Missing or damaged lane markings
* Complex intersections
* Heavy shadows
* Adverse weather conditions

Future versions of the project will explore machine-learning and segmentation-based lane detection techniques to improve robustness.

---

## Future Work

* CNN-based lane segmentation
* SCNN lane detection
* Semantic segmentation networks
* Curvature-aware path planning
* Adaptive speed control
* Traffic sign detection
* Object detection and obstacle avoidance
* End-to-end autonomous driving experiments

---

## Learning Objectives

This project was developed to gain hands-on experience with:

* Computer Vision
* Image Processing
* Vehicle Control Systems
* Autonomous Driving
* Simulation-Based Testing
* Classical Lane Detection Algorithms

It serves as a foundation for future machine-learning-based autonomous driving research.
