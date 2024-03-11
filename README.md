# IMM_LIO

**IMM-LIO (Interaction Multiple Models LiDAR-Inertial Odometry) is a reliable and effective LiDAR-inertial odometry package that supports multiple filters for estimation. It mitigates LiDAR motion distortion by leveraging high-rate IMU data through a tightly-coupled approach using the IMM filter. Three models—constant velocity, acceleration, and turning rate—are employed in this work.**

# Required Installations
- [Ubuntu 20.04](https://releases.ubuntu.com/focal/)
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [PCL >= 1.8](https://pointclouds.org/downloads/)
- [Eigen >= 3.3.4](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [ZED SDK >= 3.5](https://www.stereolabs.com/developers)
- [CUDA](https://developer.nvidia.com/cuda-downloads)

# Hardware
- Velodyne PUCK Lite
- Zed-m camera
- Handheld device
- Ground Rover Robot

  <p algin='center'>
    <img src="./doc/Lio-setup.png" alt="drawing" width="500"/>
  </p>

# System workflow
All the variables are described in detail in the [paper](https://drive.google.com/file/d/1a9Zo1jM7xKDSR00Fket8Gw3VihPOR3gR/view?usp=sharing)

  <p algin='center'>
    <img src="./doc/IMMKF_workflow.png" alt="drawing" width="800"/>
  </p>
We have designed a system that supports multiple filters, making it suitable for real-time applications. This package introduces several new features:

1. Support external for 9-axis, and 6-axis IMU.
2. The 'State Prediction' module produces multiple estimations, thanks to the use of multiple models. This ensures consistent performance even if one model degrades.
3. The 'Model Probability' module calculates the likelihood of LiDAR measurements with respect to the laser points estimated by each model in IMM. This module contributes to the final estimation and reduces computational complexity compared to direct non-linear filters.


# Install
Use the following commands to download and build the package:

```
    cd ~/caktin_ws/src    // caktin_ws or your ROS Dir
    git clone https://github.com/aralab-unr/IMM_LIO.git
    cd IMM_LIO
    cd ../..
    source devel/setup.bash
    catkin_make
```
# Prepare Rosbag for running (Velodyne or Ouster)
1. Setup LiDAR and IMU before run. To achieve optimal performance, it is essential to calibrate and synchronize both the LiDAR and IMU.
2. Edit the file ``` config/ouster64.yaml ``` or ``` config/velodyne.yaml ```
