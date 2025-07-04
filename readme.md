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



LAUNCH
run_scanmatcher.sh
run_scanmatcher_to_global_map.sh
run_graph_localizer.sh





rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_link
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link os_sensor



Instalar paquetes
numpy==1.21.6
open3d
pandas==1.5.3
gtsam
matplotlib
sudo apt install ros-noetic-ros-numpy
