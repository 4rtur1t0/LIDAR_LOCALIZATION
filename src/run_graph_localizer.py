#!/home/administrator/husky_noetic_ws/src/husky_3d_localization/.venv/bin/python3
"""
Using GTSAM in a GraphSLAM context for Localization.

We are integrating odometry, scanmatching odometry and global scanmatching.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor. Published as /localized_pose

    New nodes are created whenever a movement is found from odometry.

    Edges between nodes are created from:
    - odometry.
    - scanmatching (LiDAR) odometry.

    The scanmatcher node publishes the path as estimated by local scanmatching. This node subscribes to it and
    adds edges based on the relative estimation. In general, interpolation is used to find the correct transformations
    between the edges in the graph.

    Finally, a prior is placed on some of the edges based on a global localization node (scanmatcher to global).
    This node, uses the initial /localized_pose, computes a scanmatching and publishes the "refined estimation". Again,
    an interpolation is found to apply the prior to some to the nodes in the graph.
"""
import sys
sys.path.append('/home/administrator/husky_noetic_ws/src/husky_3d_localization/')  # Add the parent directory to the path
# sys.path.append('/home/arvc/Escritorio/SOFTWARE_ARVC_ARTURO/LOCALIZATION/LIDAR_LOCALIZATION')
from collections import deque
import rospy
import numpy as np
from graphSLAM.helper_functions import update_sm_observations, update_odo_observations, \
    filter_and_convert_gps_observations, update_prior_map_observations
from nav_msgs.msg import Odometry
from observations.gpsbuffer import GPSBuffer, GPSPosition
from observations.posesbuffer import PosesBuffer, Pose
# from sensor_msgs.msg import NavSatFix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graphSLAM.graphSLAM import GraphSLAM
import time
from config import PARAMETERS
import os
from geometry_msgs.msg import PoseWithCovarianceStamped
import tf2_ros
from geometry_msgs.msg import TransformStamped


fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('MAP/poses')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.set_title('UPDATE/OPTIMIZATION time blue (s), publication time-difference (s)')
canvas2 = FigureCanvas(fig2)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.set_title('OBSERVATIONS INDICES IN GRAPH')
canvas3 = FigureCanvas(fig3)

# the odometry input topic
ODOMETRY_TOPIC = PARAMETERS.config.get('graphslam').get('odometry_input_topic')
# the odometry scanmatching topic
ODOMETRY_SCANMATCHING_LIDAR_TOPIC = PARAMETERS.config.get('graphslam').get('odometry_scanmatching_input_topic')
# the priors on the pose as estimated by the scanmatching to map algorithm
MAP_SM_GLOBAL_POSE_TOPIC = PARAMETERS.config.get('graphslam').get('map_sm_global_pose')
# the localized estimation, based on odometry, local scanmatching and global scanmatching '/localized_pose'
OUTPUT_TOPIC = PARAMETERS.config.get('graphslam').get('localized_pose_output_topic')
# GNSS_TOPIC = '/gnss/fix_fake'
PLOTS_PATH = PARAMETERS.config.get('scanmatcher').get('plots_path')
os.makedirs(PLOTS_PATH, exist_ok=True)

# run in ros online or not
RUN_ONLINE = PARAMETERS.config.get('run_online')
# RUN_ONLINE = True

class LocalizationROSNode:
    def __init__(self):
        rospy.init_node('graph_localization_node')
        self.initial_pose_received = False
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_cb)
        rate = rospy.Rate(1)  # 10 Hz
        print('Waiting for initial pose to be set by the user in the 2D map')
        while (not self.initial_pose_received) and RUN_ONLINE:
            rate.sleep()
        if self.initial_pose_received:
            pose0 = Pose({'x': self.initial_pose.pose.pose.position.x, 'y': self.initial_pose.pose.pose.position.y,
                      'z': self.initial_pose.pose.pose.position.z, 'qx': self.initial_pose.pose.pose.orientation.x, 
                      'qy': self.initial_pose.pose.pose.orientation.y, 'qz':self.initial_pose.pose.pose.orientation.z, 
                      'qw': self.initial_pose.pose.pose.orientation.w})
            print('Initial pose set by user: ', pose0)
        else:
            pose0 = Pose({'x': 0.0, 'y': 0.0, 'z': 0.0,
                          'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0})
        # transforms
        T0 = pose0.T() #HomogeneousMatrix()
        # T LiDAR-GPS unused
        # Tlidar_gps = HomogeneousMatrix(Vector([0.36, 0, -0.4]), Euler([0, 0, 0]))
        # T LiDAR-camera unused
        # Tlidar_cam = HomogeneousMatrix(Vector([0, 0.17, 0]), Euler([0, np.pi / 2, -np.pi / 2]))
        # create the graphslam graph
        # self.graphslam = GraphSLAM(T0=T0, Tlidar_gps=Tlidar_gps, Tlidar_cam=Tlidar_cam)
        self.graphslam = GraphSLAM(T0=T0)
        self.graphslam.init_graph()

        # store odometry in deque fashion
        self.odom_buffer = PosesBuffer(maxlen=500)
        # store scanmatcher odometry in deque fashion
        self.odom_sm_buffer = PosesBuffer(maxlen=500)

        # store the priors received from the scanmatching localization node in deque fashion
        self.map_sm_prior_buffer = PosesBuffer(maxlen=500)
        self.map_sm_prior_buffer_index = deque(maxlen=500)

        # store gps readings (in utm)
        self.gps_buffer = GPSBuffer(maxlen=500)
        # store ARUCO observations and ids
        self.aruco_observations_buffer = PosesBuffer(maxlen=5000)
        self.aruco_observations_ids = deque(maxlen=5000)
        # do not skip optimization at each timestep
        self.skip_optimization = PARAMETERS.config.get('graphslam').get('skip_optimization')
        self.current_key = 0
        self.optimization_index = 1
        # graphslam times. Each node in the graph has an associated time, stored in this list
        self.graphslam_times = []
        # the initial time of the experiment
        self.start_time = None
        self.last_odom_pose = None
        # Stores the last processed index in the graph for each different observation
        self.last_processed_index = {'ODOSM': 0,
                                     'ODO': 0,
                                     'GPS': 0,
                                     'ARUCO': 0,
                                     'MAPSM': 0,
                                     'LAST_PUBLISHED_INDEX': 0}
        # Stores the indices that have been touched (put in relation)
        # for example, an ODO observation between indices (1, 2) in the graph
        self.graphslam_observations_indices = {'ODOSM': [],
                                               'ODO': [],
                                               'GPS': [],
                                               'ARUCO': [],
                                               'MAPSM': set()}
        # Load the ground truth trajectory
        # must match with the rosbag file
        # directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
        # directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO3-2025-06-16-13-49-28'
        directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
        # directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
        directory_ground_truth_path = None
        self.robotpath = PosesBuffer(maxlen=10000)
        if directory_ground_truth_path is not None:
            self.robotpath.read_data_tum(directory=directory_ground_truth_path, filename='/robot0/SLAM/data_poses_tum.txt')

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')
        print('PARAMETERS')
        print(PARAMETERS.config.get('graphslam'))

        # Subscriptions
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(ODOMETRY_SCANMATCHING_LIDAR_TOPIC, Odometry, self.odom_sm_callback)
        # rospy.Subscriber(GNSS_TOPIC, NavSatFix, self.gps_callback)
        rospy.Subscriber(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, self.map_sm_global_pose_callback)
        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(3), self.plot_timer_callback)
        # Publisher. Yes Declaring the topic to be published: /localized_pose
        
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)

        # TIME measurement
        self.update_graph_timer_callback_times = []
        self.publication_delay_times = []

        print('TOPICS:')
        print('SUBSCRIBED TO INPUT ODOMETRY TOPIC: ', ODOMETRY_TOPIC)
        print('SUBSCRIBED TO INPUT SCANMATCHING ODOMETRY TOPIC: ', ODOMETRY_SCANMATCHING_LIDAR_TOPIC)
        print('SUBSCRIBED TO INPUT SCANMATCHING TO MAP GLOBAL PRIOR: ', MAP_SM_GLOBAL_POSE_TOPIC)
        print('PUBLISHING OUTPUT ESTIMATED POSE: ', OUTPUT_TOPIC)

    def initial_pose_cb(self, data):
        """
            Set the initial pose of the robot, given by the user in the 2D map.
        """
        print('Setting initial pose: ')
        if data != None:
            self.initial_pose_received = True
            self.initial_pose = data
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
            initial_tf = TransformStamped()
            initial_tf.header.stamp = rospy.Time.now()
            initial_tf.header.frame_id = "odom"  # Typically "odom"
            initial_tf.child_frame_id = "base_link"    # Typically "base_link       
            initial_tf.transform.translation.x = data.pose.pose.position.x
            initial_tf.transform.translation.y = data.pose.pose.position.y
            initial_tf.transform.translation.z = data.pose.pose.position.z
            initial_tf.transform.rotation.x = data.pose.pose.orientation.x
            initial_tf.transform.rotation.y = data.pose.pose.orientation.y
            initial_tf.transform.rotation.z = data.pose.pose.orientation.z
            initial_tf.transform.rotation.w = data.pose.pose.orientation.w
            self.tf_broadcaster.sendTransform(initial_tf)

    def odom_callback(self, msg):
        """
            Get last odometry reading and append to buffer.
            Directly calling update_odo_observations, which should be fast at every step.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp

        pose = Pose()
        pose.from_message(msg.pose.pose)
        # In the first odometry received. add the timestamp to the list of times in the graph and also publish
        if self.last_odom_pose is None:
            self.last_odom_pose = pose
            # save initial graphslam time
            self.graphslam_times = np.array([timestamp])
            # this publishes the first node in the graph
            self.publish_graph()
            # please, do not return here, the pose has to be added to the odom_buffer
        # storing odometry buffer, but not really using it
        self.odom_buffer.append(pose, timestamp)
        start_time = time.time()
        # CAUTION! directly updating graph without  odometry buffer
        update_odo_observations(self, pose, timestamp)
        update_sm_observations(self)
        update_prior_map_observations(self)

        # odotry is at 20Hz, 1 optimization every 2 seconds: skip_optimization=40
        if self.optimization_index % self.skip_optimization == 0:
            self.graphslam.optimize()
            end_time = time.time()
            print(30*'=')
            print(f"odom_callback time after optimization :, {end_time - start_time:.4f} seconds")
            print(30 * '=')
        # Publishing the last pose after each odometry reading
        # caution: new nodes in the graph are only created whenever d_poses and th_poses is found in the odometry
        self.publish_graph()
        self.optimization_index += 1

    def odom_sm_callback(self, msg):
        """
            Get last scanmatching odometry reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.odom_sm_buffer.append(pose, timestamp)

    def map_sm_global_pose_callback(self, msg):
        """
            Store the estimations based on the map scanmatching node.
            Also store the frame_id, which corresponds to the actual index in the graph
        """
        timestamp = msg.header.stamp.to_sec()
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.map_sm_prior_buffer.append(pose, timestamp)

    def gps_callback(self, msg):
        """
            Get last GPS reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        gpsposition = GPSPosition()
        gpsposition = gpsposition.from_message(msg)
        # filter gps position
        utmposition = filter_and_convert_gps_observations(gpsposition)
        if utmposition is not None:
            self.gps_buffer.append(utmposition, timestamp)

    # def aruco_observation_callback(self, msg):
    #     """
    #         Get last odom reading and append to buffer.
    #     """
    #     timestamp = msg.header.stamp.to_sec()
    #     pose = Pose()
    #     pose.from_message(msg.pose)
    #     self.aruco_observations_buffer.append(pose, timestamp)
    #     aruco_id = int(msg.header.frame_id)
    #     self.aruco_observations_ids.append(aruco_id)

    def publish_graph(self):
        """
        Publish all the poses until now
        """
        if len(self.graphslam_times) == 0:
            print("\033[91mpublish_graph. No nodes in graph yet. Nothing to publish yet.\033[0m")
            return
        last_index = self.last_processed_index['LAST_PUBLISHED_INDEX']
        for i in range(last_index, len(self.graphslam_times)):
            T = self.graphslam.get_solution_index(i)
            if T is None:
                continue
            timestamp = self.graphslam_times[i]
            try:
                self.publish_pose(T=T, timestamp=timestamp)
            except Exception as e:
                print('Captured exception: ', e)
                pass
            self.last_processed_index['LAST_PUBLISHED_INDEX'] = i+1

    def publish_pose(self, T, timestamp):
        print('Publishing last pose:')
        position = T.pos()
        orientation = T.Q()
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(timestamp)
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]
        msg.pose.pose.position.z = position[2]
        msg.pose.pose.orientation.x = orientation.qx
        msg.pose.pose.orientation.y = orientation.qy
        msg.pose.pose.orientation.z = orientation.qz
        msg.pose.pose.orientation.w = orientation.qw
        self.pub.publish(msg)


    def plot_timer_callback(self, event):
        print('Plotting info')
        positions = self.graphslam.get_solution_positions()
        utmpositions = self.gps_buffer.get_utm_positions()
        map_sm_prior_positions = self.map_sm_prior_buffer.get_positions()
        # plot the groundtruth of the map
        map_robot_path_positions = self.robotpath.get_positions()
        # plot posittions
        ax1.clear()
        if len(positions) > 0:
            ax1.scatter(positions[:, 0], positions[:, 1], marker='.', s=30, color='blue', label='GraphSLAM solutions')
        if len(utmpositions) > 0:
            ax1.scatter(utmpositions[:, 0],
                        utmpositions[:, 1], marker='.', color='red', label='UTM readings')
        if len(map_sm_prior_positions) > 0:
            ax1.scatter(map_sm_prior_positions[:, 0],
                        map_sm_prior_positions[:, 1], marker='.', s=20, color='black', label='Map prior Scanmatching')
        if len(map_robot_path_positions) > 0:
            ax1.scatter(map_robot_path_positions[:, 0],
                        map_robot_path_positions[:, 1], marker='.', s=1, color='green', label='Map path')
        ax1.legend()

        

        canvas1.print_figure(PLOTS_PATH+'/run_graph_localizer_plot1.png', bbox_inches='tight', dpi=300)

        # plot other info
        ax2.clear()
        update_graph_timer_callback_times = np.array(self.update_graph_timer_callback_times)
        publication_delay_times = np.array(self.publication_delay_times)
        if len(update_graph_timer_callback_times):
            ax2.plot(update_graph_timer_callback_times, marker='.', color='blue')
        if len(publication_delay_times):
            ax2.plot(publication_delay_times, marker='.', color='red')
        canvas2.print_figure(PLOTS_PATH+'/run_graph_localizer_plot2.png', bbox_inches='tight', dpi=300)

        # plot the nodes in the graph
        # nodes that have been related by observations
        ax3.clear()
        if len(self.graphslam_observations_indices['ODO']):
            indices = np.array(self.graphslam_observations_indices['ODO'])
            ln = len(indices)
            ax3.plot(indices, 1.0*np.ones(ln), marker='.', color='red',
                     label='Odometry consecutive relations (only first node)')
        if len(self.graphslam_observations_indices['ODOSM']):
            indices = np.array(self.graphslam_observations_indices['ODOSM'])
            ln = len(indices)
            ax3.plot(indices, 2.0 * np.ones(ln), marker='.', color='green',
                     label='Scanmatching odometry consecutive observations')
        if len(self.graphslam_observations_indices['GPS']):
            indices = np.array(self.graphslam_observations_indices['GPS'])
            ln = len(indices)
            ax3.plot(indices, 3.0*np.ones(ln), marker='.', color='blue',
                     label='GPS prior observations')
        if len(self.graphslam_observations_indices['MAPSM']):
            indices = np.array(list(self.graphslam_observations_indices['MAPSM']))
            ln = len(indices)
            ax3.plot(indices, 4.0*np.ones(ln), marker='.', color='black',
                     label='MAP Scanmatching prior observations')
        ax3.legend()
        canvas3.print_figure(PLOTS_PATH+'/run_graph_localizer_plot3.png', bbox_inches='tight', dpi=300)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = LocalizationROSNode()
    node.run()


