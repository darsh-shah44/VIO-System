# Visual-Inertial Odometry System

Real-time camera pose estimation using monocular camera and IMU sensor fusion.

## Overview

A VIO pipeline that combines visual odometry with inertial measurements using an Extended Kalman Filter. Built in C++ with OpenCV and Eigen.

**Current Status:** In development - basic feature tracking and pose estimation working, integrating EKF for sensor fusion.

## Features

- ORB feature detection and matching
- Essential matrix-based pose estimation  
- Camera-IMU sensor fusion with EKF
- Testing on EuRoC MAV dataset

## Tech Stack

- **Languages:** C++
- **Libraries:** OpenCV, Eigen
- **Dataset:** EuRoC MAV Benchmark
- **Build System:** CMake

## Current Progress

- [x] Feature detection (ORB)
- [x] Feature matching between frames
- [x] Essential matrix computation
- [x] Pose recovery (R, t)
- [ ] EKF implementation for sensor fusion
- [ ] IMU integration
- [ ] Full trajectory tracking
- [ ] Loop closure

## Building
```bash
mkdir build && cd build
cmake .. -G "MinGW Makefiles"
cmake --build .
./vio.exe
```

## References

- EuRoC MAV Dataset
- OpenCV Documentation
- Visual-Inertial Odometry research papers

---

**Note:** This is a learning project built to understand VIO fundamentals. Currently implementing sensor fusion components.