LIDAR LOCALIZATION
A simple, yet effective, localization using a LiDAR sensor in 3D pointclouds.
Tested in semi-structured environments that include buildings and gardens.

The package includes:
- A local scanmatching node. Using Open3D, the node uses a vanilla ICP algorithm to 
estimate the trajectory of the robot.
  - A global scanmatching node. This node uses a global pcd pointcloud map that should
    have been built before (see below). In order to build this map, please use the MAP TOOLS package.
    This node gets the latest LiDAR pointcloud, gets the latest /localized_pose as published by the graph localizer, performs
    a scanmatching between the LiDAR pointcloud and the global map and publishes the result. If the initial estimation
    is approximately correct, the ICP refines the estimation and usually allows to refine it. The output is considered
    a prior on the localized pose and is also integrated by the graph localizer.
    - A graph localizer. This node estimates the robot trajectory using the
    - A node that check the timestamps of all the nodes to ensure that everything is running real time.

STEPS NEEDED TO LOCATE IN THE MAP
    - Build a map. Store in a directory.
    - Launch the ARUCO obsevations node.
    - Launch de scanmatcher node.
    - Launch the pose-map scanmatcher node.
    - Launch the run_graph_localizser node.


OTHER TOOLS
Map viewer
Timestamp viewer

data_viewer

rosparam set /use_sim_time true
rosbag play IO2-2025-03-25-16-54-17.bag --clock -r0.5 --start 0

# El nodo de localización
run_graph_localizer.sh

# el nodo de localización local
run_scanmatcher.sh

#Refina localized_pose sobre un mapa global
run_scanmatcher_to_global_map.sh

# Plotea tiempos de publicaciones importantes
./run_check_timestamps.sh


LAUNCH
run_scanmatcher.sh
run_scanmatcher_to_global_map.sh
run_graph_localizer.sh





rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom &
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link &
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link os_sensor 



Instalar paquetes
numpy==1.21.6
open3d
pandas==1.5.3
gtsam
matplotlib
sudo apt install ros-noetic-ros-numpy
