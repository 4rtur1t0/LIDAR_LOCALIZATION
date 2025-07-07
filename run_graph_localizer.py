"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
from collections import deque
import rospy
import numpy as np
from graphSLAM.helper_functions import update_sm_observations, update_odo_observations, \
    filter_and_convert_gps_observations, update_gps_observations, update_aruco_observations, \
    update_prior_map_observations
# from map.map import Map
from nav_msgs.msg import Odometry
from observations.gpsbuffer import GPSBuffer, GPSPosition
# from observations.lidarbuffer import LidarBuffer
from observations.posesbuffer import PosesBuffer, Pose
from sensor_msgs.msg import NavSatFix
# from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graphSLAM.graphSLAM import GraphSLAM
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.vector import Vector
from artelib.euler import Euler
import time

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('MAP/poses')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.set_title('UPDATE/OPTIMIZATION time blue (s), publication time-difference (s)')
canvas2 = FigureCanvas(fig2)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.set_title('OBSERVATIONS INDICES IN GRAPH')
canvas3 = FigureCanvas(fig3)

ODOMETRY_TOPIC = '/husky_velocity_controller/odom'
ODOMETRY_SCANMATCHING_LIDAR_TOPIC='/odometry_lidar_scanmatching'
# ODOMETRY_SCANMATCHING_LIDAR_TOPIC = '/genz/odometry'
# GNSS_TOPIC = '/gnss/fix'
GNSS_TOPIC = '/gnss/fix_fake'
MAP_SM_GLOBAL_POSE_TOPIC = '/map_sm_global_pose'

# the localized estimation, based on odometry, local scanmatching and global scanmatching
OUTPUT_TOPIC = '/localized_pose'

class LocalizationROSNode:
    def __init__(self):
        pose0 = Pose({'x': 0.0, 'y': 0.0, 'z': 0.0,
                      'qx': 0.00, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0})

        # pose0 = Pose({'x': 24.0, 'y': -13.05, 'z': 0.0,
        #               'qx': 0.00, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0})

        # transforms
        T0 = pose0.T() #HomogeneousMatrix()
        # T LiDAR-GPS
        Tlidar_gps = HomogeneousMatrix(Vector([0.36, 0, -0.4]), Euler([0, 0, 0]))
        # T LiDAR-camera
        Tlidar_cam = HomogeneousMatrix(Vector([0, 0.17, 0]), Euler([0, np.pi / 2, -np.pi / 2]))
        # create the graphslam graph
        self.graphslam = GraphSLAM(T0=T0, Tlidar_gps=Tlidar_gps, Tlidar_cam=Tlidar_cam)
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
        self.skip_optimization = 1
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
        # directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO4-2025-06-16-15-56-11'
        directory_ground_truth_path = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
        self.robotpath = PosesBuffer(maxlen=10000)
        self.robotpath.read_data_tum(directory=directory_ground_truth_path, filename='/robot0/SLAM/data_poses_tum.txt')

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        rospy.init_node('graph_localization_node')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')

        # Subscriptions
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(ODOMETRY_SCANMATCHING_LIDAR_TOPIC, Odometry, self.odom_sm_callback)
        rospy.Subscriber(GNSS_TOPIC, NavSatFix, self.gps_callback)
        # the ARUCO observations
        # rospy.Subscriber('/aruco_observation', PoseStamped, self.aruco_observation_callback)
        rospy.Subscriber(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, self.map_sm_global_pose_callback)

        # Set up a timer to periodically update the graphSLAM graph
        # rospy.Timer(rospy.Duration(1), self.optimize_graph_timer_callback)
        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(3), self.plot_timer_callback)

        # Publisher
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)

        # TIME measurement
        self.update_graph_timer_callback_times = []
        self.publication_delay_times = []

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
        if self.last_odom_pose is None:
            self.last_odom_pose = pose
            # save initial graphslam time
            self.graphslam_times = np.array([timestamp])

        # storing odometry buffer, but not really using it
        self.odom_buffer.append(pose, timestamp)
        start_time = time.time()
        # CAUTION! directly updating graph without  odometry buffer
        update_odo_observations(self, pose, timestamp)
        update_sm_observations(self)
        update_prior_map_observations(self)

        # odotry is at 20Hz, 1 optimization every 2 seconds
        if self.optimization_index % 50 == 0:
            self.graphslam.optimize()


        # performing all the rest of observations in the same thread
        # update_sm_observations(self)
        # update_prior_map_observations(self)
        # caution, publishing here!
        # CAUTION! DO NOT OPTIMIZE IN THIS FUNCITON
        self.publish_graph()
        self.optimization_index += 1
        end_time = time.time()
        # print(30*'=')
        # print(f"odom_callback time time:, {end_time - start_time:.4f} seconds")
        # print(30 * '=')
        # self.update_graph_timer_callback_times.append(end_time - start_time)

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

    def aruco_observation_callback(self, msg):
        """
            Get last odom reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        pose = Pose()
        pose.from_message(msg.pose)
        self.aruco_observations_buffer.append(pose, timestamp)
        aruco_id = int(msg.header.frame_id)
        self.aruco_observations_ids.append(aruco_id)

    # def optimize_graph_timer_callback(self, event):
    #     start_time = time.time()
    #     # performing all the rest of observations in the same thread
    #     update_sm_observations(self)
    #     update_prior_map_observations(self)
    #     self.graphslam.optimize()
    #     end_time = time.time()
    #     print(30 * '=')
    #     print(f"optimize_graph_timer_callback time time:, {end_time - start_time:.4f} seconds")
    #     print(30 * '=')

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
        position = T.pos()# print(pose)
        orientation = T.Q()
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(timestamp) #rospy.Time.now()
        # caution: the frame_id stores the index in the graph, so that the graph localizer in map can
        # also find a localization in the map and return.
        msg.header.frame_id = "map" #str(index_in_graph) #"map"
        msg.child_frame_id = "odom"
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]
        msg.pose.pose.position.z = position[2]
        # q = pose.rotation().toQuaternion()
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
        # map_robot_path_positions = map_robot_path_positions[0:1500, :]
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
            # N = len(positions)
            # map_robot_path_positions = map_robot_path_positions[0:N]
            ax1.scatter(map_robot_path_positions[:, 0],
                        map_robot_path_positions[:, 1], marker='.', s=1, color='green', label='Map path')
        ax1.legend()
        canvas1.print_figure('plots/run_graph_localizer_plot1.png', bbox_inches='tight', dpi=300)

        # plot other info
        ax2.clear()
        update_graph_timer_callback_times = np.array(self.update_graph_timer_callback_times)
        publication_delay_times = np.array(self.publication_delay_times)
        if len(update_graph_timer_callback_times):
            ax2.plot(update_graph_timer_callback_times, marker='.', color='blue')
        if len(publication_delay_times):
            ax2.plot(publication_delay_times, marker='.', color='red')
        canvas2.print_figure('plots/run_graph_localizer_plot2.png', bbox_inches='tight', dpi=300)

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
        canvas3.print_figure('plots/run_graph_localizer_plot3.png', bbox_inches='tight', dpi=300)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    filename = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17.bag'
    node = LocalizationROSNode()
    node.run()


