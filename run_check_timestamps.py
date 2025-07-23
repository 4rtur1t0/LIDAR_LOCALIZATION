"""
    A node that subscribes to the important topics and prints a
    plot (png, in /plots) with the last published times. For debugging purposes.
"""
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, PointCloud2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from config import PARAMETERS

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)

# subscribing to these topics and checking times
ODOMETRY_TOPIC = PARAMETERS.config.get('graphslam').get('odometry_input_topic')
# CAUTION: this topic must be subscribed to the /ouster/points (high rate) topic
POINTCLOUD_TOPIC = PARAMETERS.config.get('scanmatcher').get('pointcloud_input_topic')
ODOMETRY_SCANMATCHING_TOPIC = PARAMETERS.config.get('scanmatcher').get('odometry_output_topic')
PRIOR_SM_GLOBAL_MAP_POSE = PARAMETERS.config.get('graphslam').get('map_sm_global_pose')
# ESTIMATED POSE, this is the output of the run_graph_localizer algorithm
LOCALIZED_POSE = PARAMETERS.config.get('graphslam').get('localized_pose_output_topic')


class CheckTimestampsNode:
    def __init__(self):
        self.start_time = None
        self.odometry_times = []
        self.pcd_times = []
        self.odometry_scanmatching_times = []
        self.localized_pose_times = []
        self.prior_sm_gloal_map_pose_times = []

        print('Initializing check times node!')
        rospy.init_node('check_times_node')
        print('Subscribing to ODOMETRY, scanmatching, scanmatching global and pointclouds')
        print('ODOMETRY TOPIC: ', ODOMETRY_TOPIC)
        print('ODOMETRY SCANMATCHING TOPIC: ', ODOMETRY_SCANMATCHING_TOPIC)
        print('POINTCLOUD TOPIC: ', POINTCLOUD_TOPIC)
        print('SCANMATCHING TO MAP TOPIC: ', PRIOR_SM_GLOBAL_MAP_POSE)
        print('GRAPH ESTIMATED POSE GTSAM: ', LOCALIZED_POSE)

        # Subscriptions to the TOPICS
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odometry_callback)
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)
        rospy.Subscriber(ODOMETRY_SCANMATCHING_TOPIC, Odometry, self.odometry_scanmatching_callback)
        rospy.Subscriber(PRIOR_SM_GLOBAL_MAP_POSE, Odometry, self.prior_sm_gloal_map_pose_callback)
        rospy.Subscriber(LOCALIZED_POSE, Odometry, self.localized_pose_callback)

        # Set up a timer to periodically update the graph
        rospy.Timer(rospy.Duration(5), self.plot_timer_callback)

    def odometry_callback(self, msg):
        """
            Get last odometry timestamp.
            Robot odometry.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.odometry_times.append(timestamp)

    def odometry_scanmatching_callback(self, msg):
        """
            Get last scanmatching timestamp.ç
            published by the run_scanmatcher.py node.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.odometry_scanmatching_times.append(timestamp)

    def pc_callback(self, msg):
        """
        Get last pcd timestamp.
        published by the lidar
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.pcd_times.append(timestamp)

    def localized_pose_callback(self, msg):
        """
            Obtain the last timestamp on the estimated pose on the robot path
            published by run_graph_localizer.py.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.localized_pose_times.append(timestamp)

    def prior_sm_gloal_map_pose_callback(self, msg):
        """
            Obtain the last timestamp on the estimations against the global pcd map.
            (run_scanmatcher_to_global_map.py)
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.prior_sm_gloal_map_pose_times.append(timestamp)

    def plot_timer_callback(self, event):
        print('Plotting info')
        delta_see = 1
        last_data = 100
        ax1.clear()
        odometry_times = np.array(self.odometry_times)-self.start_time
        if len(odometry_times) > 0:
            max_len = min(len(odometry_times), last_data)
            odometry_times = odometry_times[-max_len:]
            ax1.plot(odometry_times, marker='.', color='red', label='Tiempos odometria')
        odometry_scanmatching_times = np.array(self.odometry_scanmatching_times) - self.start_time - delta_see
        if len(odometry_scanmatching_times) > 0:
            max_len = min(len(odometry_scanmatching_times), last_data)
            odometry_scanmatching_times = odometry_scanmatching_times[-max_len:]
            ax1.plot(odometry_scanmatching_times, marker='.', color='green', label='Tiempos local scanmatching')
        pcd_times = np.array(self.pcd_times)-self.start_time
        if len(pcd_times) > 0:
            max_len = min(len(pcd_times), last_data)
            pcd_times = pcd_times[-max_len:]
            ax1.plot(pcd_times, marker='.', color='blue', label='Pointcloud times')
        localized_pose_times = np.array(self.localized_pose_times) - self.start_time
        if len(localized_pose_times) > 0:
            max_len = min(len(localized_pose_times), last_data)
            localized_pose_times = localized_pose_times[-max_len:]
            ax1.plot(localized_pose_times, marker='.', color='black', label='Localized pose times (LOCALIZATION)')

        prior_sm_gloal_map_pose_times = np.array(self.prior_sm_gloal_map_pose_times) - self.start_time
        if len(prior_sm_gloal_map_pose_times) > 0:
            max_len = min(len(prior_sm_gloal_map_pose_times), last_data)
            prior_sm_gloal_map_pose_times = prior_sm_gloal_map_pose_times[-max_len:]
            ax1.plot(prior_sm_gloal_map_pose_times, marker='.', color='yellow', label='Map priors GLOBAL MAP scanmatching')

        ax1.legend()
        ax1.grid()
        canvas1.print_figure('plots/run_check_timestamps_plot1_times.png', bbox_inches='tight', dpi=300)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = CheckTimestampsNode()
    node.run()


