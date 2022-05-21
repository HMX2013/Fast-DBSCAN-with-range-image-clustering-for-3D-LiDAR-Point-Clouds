cmake_minimum_required(VERSION 3.0.2)
project(range_image_segment)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf2_ros
  tf2_geometry_msgs
  dynamic_reconfigure
  autoware_msgs
  jsk_recognition_msgs
)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "-O2 -Wall ${CMAKE_CXX_FLAGS}")

generate_dynamic_reconfigure_options(
  cfg/range_image_segment.cfg
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES range_image_segment
  CATKIN_DEPENDS roscpp std_msgs
  # DEPENDS system_lib
)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/dbscan_kdtree.hpp
# )

## Add cmake target dependencies of the library
## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(range_image_segment 
  src/range_image_segment.cpp
  src/cluster.cpp
)

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(range_image_segment ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(range_image_segment
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)