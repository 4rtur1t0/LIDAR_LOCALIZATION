"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
from collections import deque
import rospy
import numpy as np
from graphSLAM.helper_functions import update_sm_observations, update_odo_observations, \
    filter_and_convert_gps_observations, update_gps_observations, update_aruco_observations
from map.map import Map
from nav_msgs.msg import Odometry
from observations.gpsbuffer import GPSBuffer, GPSPosition
from observations.posesbuffer import PosesBuffer, Pose
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graphSLAM.graphSLAM import GraphSLAM
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.vector import Vector
from artelib.euler import Euler
from config import PARAMETERS
from tools.gpsconversions import gps2utm

fig, ax = plt.subplots()
canvas = FigureCanvas(fig)


class LocalizationROSNode:
    def __init__(self):
        print('Initializing localization node!')
        rospy.init_node('localization_node')
        print('Subscribing to ODOMETRY, GNSS')
        print('WAITING FOR MESSAGES!')
        # Subscriptions
        rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/odometry_lidar_scanmatching', Odometry, self.odom_sm_callback)
        rospy.Subscriber('/gnss/fix', NavSatFix, self.gps_callback)
        # the ARUCO observations
        rospy.Subscriber('/aruco_observation', PoseStamped, self.aruco_observation_callback)
        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(2), self.plot_timer_callback)
        # Set up a timer to periodically update the graph
        rospy.Timer(rospy.Duration(2), self.update_graph_timer_callback)
        # Publisher
        self.pub = rospy.Publisher('/localized_pose', Odometry, queue_size=10)
        # transforms
        T0 = HomogeneousMatrix()
        # T LiDAR-GPS
        Tlidar_gps = HomogeneousMatrix(Vector([0.36, 0, -0.4]), Euler([0, 0, 0]))
        # T LiDAR-camera
        Tlidar_cam = HomogeneousMatrix(Vector([0, 0.17, 0]), Euler([0, np.pi / 2, -np.pi / 2]))
        # create the graphslam graph
        self.graphslam = GraphSLAM(T0=T0, Tlidar_gps=Tlidar_gps, Tlidar_cam=Tlidar_cam)
        self.graphslam.init_graph()

        # store odometry in deque fashion
        self.odom_buffer = PosesBuffer(maxlen=5000)
        # store scanmatcher odometry in deque fashion
        self.odom_sm_buffer = PosesBuffer(maxlen=5000)
        # store gps readings (in utm)
        self.gps_buffer = GPSBuffer(maxlen=1000)
        # store ARUCO observations and ids
        self.aruco_observations_buffer = PosesBuffer(maxlen=5000)
        self.aruco_observations_ids = deque(maxlen=5000)

        self.skip_optimization = 30
        self.current_key = 0
        self.optimization_index = 1
        # gparhslam times. Each node in the graph has an associated time
        self.graphslam_times = []
        self.start_time = None

        # LOAD THE MAP
        directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
        self.map = Map()
        self.map.read_data(directory=directory)

    def odom_callback(self, msg):
        """
            Get last odometry reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.odom_buffer.append(pose, timestamp)

    def odom_sm_callback(self, msg):
        """
            Get last scanmatching odometry reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
            self.graphslam_times = np.array([timestamp])
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.odom_sm_buffer.append(pose, timestamp)

    def gps_callback(self, msg):
        """
            Get last GPS reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
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
        if self.start_time is None:
            self.start_time = timestamp
        pose = Pose()
        pose.from_message(msg.pose)
        self.aruco_observations_buffer.append(pose, timestamp)
        aruco_id = int(msg.header.frame_id)
        self.aruco_observations_ids.append(aruco_id)

    def update_graph_timer_callback(self, event):
        """
        Called at a fixed timestamp to integrate new observations in the graph
        """
        print('UPDATE OBSERVATIONS!! SM, ODO, GPS')
        update_sm_observations(self)
        update_odo_observations(self)
        update_gps_observations(self)
        update_aruco_observations(self)

        self.optimization_index += 1
        if self.optimization_index % self.skip_optimization == 0:
            print(300*'+')
            print('Optimize Graph!!')
            print(300 * '+')
            self.graphslam.optimize()
        else:
            print('Skipping optimization')
            print('self.optimization index: ', self.optimization_index)



    def publish_pose(self, T):
        if T is None:
            return
        print('Publishing last pose:')
        position = T.pos()# print(pose)
        orientation = T.Q()
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
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
        ax.clear()
        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], marker='.', color='blue')
        # if len(self.utm_valid_positions) > 0:
        #     utm_valid_positions = np.array(self.utm_valid_positions)
        #     ax.scatter(utm_valid_positions[:, 0],
        #                utm_valid_positions[:, 1], marker='.', color='red')
        canvas.print_figure('plot.png', bbox_inches='tight')



    def run(self):
        rospy.spin()


if __name__ == "__main__":
    filename = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17.bag'
    node = LocalizationROSNode()
    node.run()


