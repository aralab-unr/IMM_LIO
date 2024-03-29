# IMM_LIO

**IMM-LIO (Interaction Multiple Models LiDAR-Inertial Odometry) is a reliable and effective LiDAR-inertial odometry package that supports multiple filters for estimation. It mitigates LiDAR motion distortion by leveraging high-rate IMU data through a tightly-coupled approach using the IMM filter. Three models—constant velocity, acceleration, and turning rate—are employed in this work.**
<p align='center'>
    <img src="./doc/New_college.gif" alt="drawing" width="800"/>
</p>

# Dependencies
The framework has been tested with ROS Noetic and Ubuntu 20.04. The following configuration, along with the required dependencies, has been verified for compatibility:

- [Ubuntu 20.04](https://releases.ubuntu.com/focal/)
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) (```roscpp```, ```std_msgs```, ```sensor_msgs```, ```geometry_msgs```)
- C++ 14
- [PCL >= 1.8](https://pointclouds.org/downloads/)
- [Eigen >= 3.3.4](http://eigen.tuxfamily.org/index.php?title=Main_Page)
- [ZED SDK >= 3.5](https://www.stereolabs.com/developers)
- [CUDA](https://developer.nvidia.com/cuda-downloads) (Recommend to use CUDA toolkit >= 11 for Ubuntu 20.04)

# Hardware
- [Velodyne PUCK Lite](https://velodynelidar.com/products/puck-lite/) (360 HFOV, +15 degree VFOV, 100m range, 10Hz) 
- [Zedm camera](https://www.stereolabs.com/developers/release)
- Handheld device
- Ground Rover Robot

<p align='center'>
    <img src="./doc/Lio-setup.png" alt="drawing" width="600"/>
</p>

# System workflow
All the variables are described in detail in the [paper](https://drive.google.com/file/d/1a9Zo1jM7xKDSR00Fket8Gw3VihPOR3gR/view?usp=sharing)

<p align='center'>
    <img src="./doc/IMMKF_workflow.png" alt="drawing" width="700"/>
</p>
We have designed a system that supports multiple filters, making it suitable for real-time applications. This package introduces several new features:

1. Support external for 9-axis, and internal 6-axis IMU.
2. The 'State Prediction' module produces multiple estimations, thanks to the use of multiple models. This ensures consistent performance even if one model degrades.
3. The 'Model Probability' module calculates the likelihood of LiDAR measurements with respect to the laser points estimated by each model in IMM. This module contributes to the final estimation and reduces computational complexity compared to direct non-linear filters.


# Install
Use the following commands to download and build the package: (The code is implemented in ROS1)

```
    mkdir catkin_ws
    cd ~/caktin_ws/src    // caktin_ws or your ROS Dir
    git clone https://github.com/aralab-unr/IMM_LIO.git
    cd IMM_LIO
    cd ../..
    source devel/setup.bash
    catkin_make
```
# Prepare Rosbag for running 
Requires an input LiDAR point cloud of type ```sensor_msgs::PointCloud2``` and IMU input of type ```sensor_msgs::IMU```

1. Setup LiDAR and IMU before run. To achieve optimal performance, it is essential to calibrate and synchronize both the LiDAR and IMU.
2. Edit the file ``` config/velodyne.yaml ``` to set the parameters.
3. Set the LiDAR and IMU topic at: ```lid_topic```, ```imu_topic```
4. Change the LiDAR, and IMU extrinsic calibration parameters: ``` extrinsic_R ``` , and ``` extrinsic_T ``` .
5. Set the IMU as base frame.

# Run the package
For Velodyne type
1. Run the launch file:
```
    roslaunch imm_lio velodyne.launch

2. Play existing bag files:
```
    rosbag play your-file.bag

3. Download [sample dataset](https://drive.google.com/drive/folders/1Bxe2sPL9lQXFsh6_xb5OAr8OxKFyTGON?usp=drive_link) which are collected in UNR campuse to test the package. In these dataset, the point cloud topic is ``` "/velodyne_points"```, and the imu topic need to be set to ``` "zed/zed_nodelet/imu/data"```
4. Download Urban Hong Kong dataset [medium-urban](https://www.dropbox.com/s/mit5v1yo8pzh9xq/UrbanNav-HK_TST-20210517_sensors.bag?e=1&dl=0) [deep-urban](https://www.dropbox.com/s/1g3dllvdrgihkij/UrbanNav-HK_Whampoa-20210521_sensors.bag?e=1&dl=0). Set the imu topic to ```"/imu/data" ```.

<p align='center'>
    <img src="./doc/SEM_GT.png" alt="drawing" width="200"/>
    <img src="./doc/SEM_IMM.png" alt="drawing" width="200"/>
    <img src="./doc/Urban-GT.png" alt="drawing" width="200"/>
    <img src="./doc/UrBan_IMM.png" alt="drawing" width="200"/>
</p>

<p align='center'>
    <img src="./doc/UNR_SEM.gif" alt="drawing" width="400"/>
    <img src="./doc/urban.gif" alt="drawing" width="400"/>
</p>

Depending on the number of models and their characteristics, the users can modify the transition matrix of the IMM model (p_ij) in the 'IMM_lio.cpp' file. The default values used in this work are [0.9, 0.01, 0.09; 0.025, 0.75, 0.225; 0.075, 0.175, 0.75]. Ensure that the sum of each row equals 1

# Related Package
- [LiDAR-IMU calibration](https://github.com/hku-mars/LiDAR_IMU_Init)

- [Fast-LIO2](https://github.com/hku-mars/FAST_LIO?tab=readme-ov-file)

# Acknowledgement
We thank the authors of Fast-LIO2 and [LiDAR-IMU calibration](https://github.com/hku-mars/LiDAR_IMU_Init) for providing open-source packages:

- Xu, W., Cai, Y., He, D., Lin, J., & Zhang, F. (2022). Fast-lio2: Fast direct lidar-inertial odometry. IEEE Transactions on Robotics, 38(4), 2053-2073.

- F. Zhu, Y. Ren and F. Zhang, "Robust Real-time LiDAR-inertial Initialization," 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Kyoto, Japan, 2022, pp. 3948-3955, doi: 10.1109/IROS47612.2022.9982225.

# Contact
- [An Nguyen](mailto:anguyenduy@nevada.unr.edu)
- [Hung La](mailto:hla@unr.edu)
