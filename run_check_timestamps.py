"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.


    This node subscribes to the localized_pose topic.
    The localized_pose topic is initially published by the localization node.
    The initial pose is used to find a number of close pointclouds in the map. A registration is then performed
    As a result, we end up having another prior3Dfactor observation on the state X(i)

"""
# from collections import deque
import rospy
import numpy as np
# from graphSLAM.helper_functions import update_sm_observations, update_odo_observations, \
#     filter_and_convert_gps_observations, update_gps_observations, update_aruco_observations
# from map.map import Map
from nav_msgs.msg import Odometry
# from observations.gpsbuffer import GPSBuffer, GPSPosition
# from observations.lidarbuffer import LidarBuffer, LidarScan
# from observations.posesbuffer import PosesBuffer, Pose
# from scanmatcher.scanmatcher import ScanMatcher
from sensor_msgs.msg import NavSatFix, PointCloud2
# from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from graphSLAM.graphSLAM import GraphSLAM
# from artelib.homogeneousmatrix import HomogeneousMatrix
# from artelib.vector import Vector
# from artelib.euler import Euler
# from config import PARAMETERS
# from tools.gpsconversions import gps2utm

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)


ODOMETRY_TOPIC = '/husky_velocity_controller/odom'
# CAUTION: this topic must be subscribed to the /ouster/points (high rate) topic
POINTCLOUD_TOPIC = '/ouster/points_low_rate'
ODOMETRY_SCANMATCHING_TOPIC = '/odometry_lidar_scanmatching'
PRIOR_SM_GLOBAL_MAP_POSE = '/map_sm_global_pose'
# INITIAL ESTIMATION POSE, this is the output of the run_graph_localizer algorithm
LOCALIZED_POSE = '/localized_pose'



class CheckTimestampsNode:
    def __init__(self):
        self.start_time = None
        self.odometry_times = []
        self.pcd_times = []
        self.odometry_scanmatching_times = []
        self.localized_pose_times = []
        self.prior_sm_gloal_map_pose_times = []

        # self.gnss_times =  []
        print('Initializing check times node!')
        rospy.init_node('check_times_node')
        print('Subscribing to ODOMETRY and pointclouds')

        # Subscriptions to the pointcloud topic and to the
        # current localized pose
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odometry_callback)
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)
        rospy.Subscriber(ODOMETRY_SCANMATCHING_TOPIC, Odometry, self.odometry_scanmatching_callback)
        rospy.Subscriber(PRIOR_SM_GLOBAL_MAP_POSE, Odometry, self.prior_sm_gloal_map_pose_callback)
        rospy.Subscriber(LOCALIZED_POSE, Odometry, self.localized_pose_callback)

        # Set up a timer to periodically update the graph
        rospy.Timer(rospy.Duration(5), self.plot_timer_callback)

    def odometry_callback(self, msg):
        """
            Get last odometry reading and append to buffer.
            Directly calling update_odo_observations, which should be fast at every step.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        # storing odometry buffer, but not really using it
        self.odometry_times.append(timestamp)

    def odometry_scanmatching_callback(self, msg):
        """
            Get last scanmatching odometry reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.odometry_scanmatching_times.append(timestamp)

    def pc_callback(self, msg):
        """
        Get last pcd reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.pcd_times.append(timestamp)

    def localized_pose_callback(self, msg):
        """
            Obtain the last estimations on the robot path
            This /localized_pose is used as an initial estimation and usually obtained from the localization node itself.
            This should be the /localized_pose topic, which maintains
            the last localization with all the information, excluding
            the localization with respec to the map.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.localized_pose_times.append(timestamp)

    def prior_sm_gloal_map_pose_callback(self, msg):
        """
            Obtain the last estimations on the robot path
            This /localized_pose is used as an initial estimation and usually obtained from the localization node itself.
            This should be the /localized_pose topic, which maintains
            the last localization with all the information, excluding
            the localization with respec to the map.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.prior_sm_gloal_map_pose_times.append(timestamp)

    def plot_timer_callback(self, event):
        print('Plotting info')
        ax1.clear()
        odometry_times = np.array(self.odometry_times)-self.start_time
        if len(odometry_times) > 0:
            ax1.plot(odometry_times, marker='.', color='red', label='Tiempos odometria')
        odometry_scanmatching_times = np.array(self.odometry_scanmatching_times) - self.start_time
        if len(odometry_scanmatching_times) > 0:
            ax1.plot(odometry_scanmatching_times, marker='.', color='green', label='Tiempos local scanmatching')
        pcd_times = np.array(self.pcd_times)-self.start_time
        if len(pcd_times) > 0:
            ax1.plot(pcd_times, marker='.', color='blue', label='Pointcloud times')
        localized_pose_times = np.array(self.localized_pose_times) - self.start_time
        if len(localized_pose_times) > 0:
            ax1.plot(localized_pose_times, marker='.', color='black', label='Localized pose times (LOCALIZATION)')

        prior_sm_gloal_map_pose_times = np.array(self.prior_sm_gloal_map_pose_times) - self.start_time
        if len(prior_sm_gloal_map_pose_times) > 0:
            ax1.plot(prior_sm_gloal_map_pose_times, marker='.', color='yellow', label='Map priors GLOBAL MAP scanmatching')

        ax1.legend()
        ax1.grid()
        canvas1.print_figure('plots/run_check_timestamps_plot1_times.png', bbox_inches='tight', dpi=300)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = CheckTimestampsNode()
    node.run()


