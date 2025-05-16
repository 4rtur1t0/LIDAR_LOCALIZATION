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
from observations.lidarbuffer import LidarBuffer, LidarScan
from observations.posesbuffer import PosesBuffer, Pose
from scanmatcher.scanmatcher import ScanMatcher
from sensor_msgs.msg import NavSatFix, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graphSLAM.graphSLAM import GraphSLAM
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.vector import Vector
from artelib.euler import Euler
from config import PARAMETERS
from tools.gpsconversions import gps2utm

fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvas(fig)

POINTCLOUD_TOPIC = '/ouster/points_low_rate'
# WE ARE PUBLISHING THE PRIOR ESTIMATION HERE, as a result of the localization in the map
MAP_SM_GLOBAL_POSE_TOPIC = '/map_sm_global_pose'

# INITIAL ESTIMATION POSE, this is the output of the run_graph_localizer algorithm
INITIAL_ESTIMATED_POSE = '/localized_pose'

class GlobalScanMatchingROSNode:
    def __init__(self):
        self.start_time = None
        # the lidar buffer
        self.pcdbuffer = LidarBuffer(maxlen=5)
        self.times_lidar = []
        # store odometry in deque fashion
        self.initial_poses_buffer = PosesBuffer(maxlen=5000)
        # this stores the indices in graphslam
        self.initial_poses_indices = []

        # LOAD THE MAP
        print('Loading MAP')
        directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
        self.map = Map()
        self.map.read_data(directory=directory)

        # the scanmatching object
        self.scanmatcher = ScanMatcher()

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        rospy.init_node('localization_node')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')

        # Subscriptions
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)
        rospy.Subscriber(INITIAL_ESTIMATED_POSE, Odometry, self.initial_pose_callback)

        # Set up a timer to periodically update the graph
        rospy.Timer(rospy.Duration(1), self.compute_global_scanmatching)
        rospy.Timer(rospy.Duration(1), self.plot_timer_callback)
        # Publisher
        self.pub = rospy.Publisher(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, queue_size=10)

        self.prior_estimations = []
        self.processed_map_poses = []

    def pc_callback(self, msg):
        """
        Get last pcd reading and append to buffer.

        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.times_lidar.append(timestamp)
        print('Received pointcloud')
        # print(50*'=')
        # print('Appending pointcloud')
        # print(50*'=')
        pcd = LidarScan(time=timestamp, pose=None)
        pcd.load_pointcloud_from_msg(msg=msg)
        self.pcdbuffer.append(pcd=pcd, time=timestamp)
        print(30 * '+')
        return

    def initial_pose_callback(self, msg):
        """
            Store the last estimations on the robot path
        """
        timestamp = msg.header.stamp.to_sec()
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.initial_poses_buffer.append(pose, timestamp)
        self.initial_poses_indices.append(int(msg.header.frame_id))


    def compute_global_scanmatching(self, event):
        """
            Iterate through the pcds and estimate relative measurements
            Temptative, initial,
            - Find the current pointcloud and the closest pointcloud in the map.
            - Find the corresponding pose in the map.
            - Compute the relative transformation (Tij0=I... I+delta)
            - Compute the prior information and plot on the map.

            final, should be:
            - look for last the current estimation of the pose.
            - look for nearby poses in the map.
            - compute initial estimation.
        """
        # max_proc = 2
        # for i in range(max_proc):
        i = 0
        if len(self.pcdbuffer) == 0:
            return
        current_pcd = self.pcdbuffer[i]
        current_pcd_time = self.pcdbuffer.times[i]

        current_pcd_time = current_pcd_time + 0.5

        # current_time = rospy.Time.now().to_sec()
        # diff = current_time-current_pcd_time
        # current test approach: find a map pose closest in time
        # needed approach: get pointclouds in the map closest in euclidean distance
        map_pose, time_map = self.map.get_pose_closest_to_time(timestamp=current_pcd_time, delta_threshold_s=1.0)
        if map_pose is None:
            return
            # continue
        # get the closest pcd in the map
        map_pcd, pointcloud_time = self.map.get_pcd_closest_to_time(timestamp=current_pcd_time, delta_threshold_s=1.0)
        diff = current_pcd_time-pointcloud_time
        print('Diff in time: current pcd and map pcd: ', diff)

        # process the current pcd
        current_pcd.down_sample(voxel_size=None)
        current_pcd.filter_points()
        current_pcd.estimate_normals()
        # current_pcd.draw_cloud()

        # current the map pcd
        map_pcd.load_pointcloud()
        map_pcd.down_sample(voxel_size=None)
        map_pcd.filter_points()
        map_pcd.estimate_normals()
        # map_pcd.draw_cloud()
        # now, in this, test, consider that the initial transformation is the identity
        # In the final approach: consider that the relative initial transformation is known
        # Tij0 = HomogeneousMatrix(Vector([0.1, 0.1, 0]), Euler([0.1, 0.1, 0.1]))
        Tij0 = HomogeneousMatrix()
        Tij = self.scanmatcher.registration(current_pcd, map_pcd, Tij_0=Tij0, show=False)

        map_pcd.unload_pointcloud()
        # the map pose (pointcloud)
        T0j = map_pose.T()
        # estimate the initial i
        T0i = T0j*Tij.inv()
        self.prior_estimations.append(T0i)
        self.processed_map_poses.append(map_pose.T())
        # remove pcd from the list
        # self.pcdbuffer.popleft()


    def publish_prior_information_pose(self, T):
        """
        Publish the estimation found on any pose close to the published time.
        """
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
        prior_estimations = []
        processed_map_poses = []
        for T0i in self.prior_estimations:
            prior_estimations.append(T0i.pos())
        prior_estimations = np.array(prior_estimations)

        for T0j in self.processed_map_poses:
            processed_map_poses.append(T0j.pos())
        processed_map_poses = np.array(processed_map_poses)

        ax.clear()
        if len(prior_estimations) > 0:
            ax.scatter(prior_estimations[:, 0], prior_estimations[:, 1], marker='.', color='blue')

        if len(processed_map_poses) > 0:
            ax.scatter(processed_map_poses[:, 0],
                       processed_map_poses[:, 1], marker='.', color='red')

        canvas.print_figure('plot.png', bbox_inches='tight', dpi=300)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    filename = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17.bag'
    node = GlobalScanMatchingROSNode()
    node.run()


