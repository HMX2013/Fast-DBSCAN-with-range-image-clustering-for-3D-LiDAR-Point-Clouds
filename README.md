# range_image_segment
A ROS package to perform `range_image_segment` for 3D lidar pointcloud.

# reference
* Two-Layer-Graph Clustering for Real-Time 3D LiDAR Point Cloud Segmentation
* Efficient Online Segmentation for Sparse 3D Laser Scans
* Line Extraction in 2D Range Images for Mobile Robotics
* The code from the LeGO-LOAM SLAM algorithm

## Requirement
* pcl 1.8
* ros kinetic or melodic

## Run
$ catkin_make
$ source ./devel/setup.bash
$ roslaunch range_image_segment range_image_segment.launch