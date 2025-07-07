"""
A simple Scanmatcher for LiDAR pointclouds.
Publishes a global estimated pose based on the first odometry reading found.
No GPS or IMU is integrated.
Uses Open3d for ICP

"""
import rospy
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from observations.lidarbuffer import LidarBuffer, LidarScan
from observations.posesbuffer import Pose
from config import PARAMETERS
from observations.posesbuffer import PosesBuffer
# from tools.gpsconversions import gps2utm
from sensor_msgs.msg import PointCloud2
from scanmatcher.scanmatcher import ScanMatcher
import time


ODOMETRY_TOPIC = '/husky_velocity_controller/odom'
POINTCLOUD_TOPIC = '/ouster/points_low_rate'
# the output estimation
OUTPUT_TOPIC = '/odometry_lidar_scanmatching'

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)

# fig3, ax3 = plt.subplots(figsize=(6, 4))
# ax3.set_title('OBSERVATIONS INDICES IN GRAPH')
# canvas3 = FigureCanvas(fig3)


def compute_rel_distance(odo1, odo2):
    Ti = odo1.T()
    Tj = odo2.T()
    Tij = Ti.inv() * Tj
    d = np.linalg.norm(Tij.pos())
    e1 = np.linalg.norm(Tij.euler()[0].abg)
    e2 = np.linalg.norm(Tij.euler()[1].abg)
    theta = min(e1, e2)
    print('Relative (d, theta):', d, theta)
    return d, theta


class ScanmatchingNode:
    def __init__(self):
        print('Initializing local scanmatching node!')
        rospy.init_node('scanmatching_node')
        print('Subscribing to ODOMETRY and pointclouds')
        print('CAUTION: odometry and poinclouds are synchronized with filter messages')
        print('WAITING FOR MESSAGES!')
        # Subscriptions
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback, queue_size=10)
        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(5), self.timer_callback_plot_info)

        # Publisher
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)
        # stores odometry poses as a short buffer with deque
        self.odombuffer = PosesBuffer(maxlen=1000)
        # the lidar buffer
        # self.pcdbuffer = LidarBuffer(maxlen=300)
        # store the results from the beginning of the experiment
        self.relative_transforms = []
        self.global_transforms = []
        self.last_global_transform_published = 0
        self.start_time = None

        self.times_odometry = []
        self.times_lidar = []
        # self.positions_sm = []
        # the scanmatching object
        self.scanmatcher = ScanMatcher()
        self.timer_callback_process_scanmatching_computation_time = []
        rospy.loginfo("ScanMatcher with odom/pc running.")

        # the two pcds used to compute the relative transformation
        self.pcd1 = None
        self.pcd2 = None
        # the global transforamtion
        self.Tg = HomogeneousMatrix()

        # lag
        self.computation_times = []
        self.frequency = []

        self.time_diffs = []
        self.time_sleep_synchro = 1.5


    def run(self):
        rospy.spin()

    def odom_callback(self, msg):
        """
        Get last odom reading and append to buffer.
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.times_odometry.append(timestamp)
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.odombuffer.append(pose, timestamp)

    def pc_callback(self, msg):
        """
        Get last pcd reading.
        Next, perform scanmatching with the last received pcd.
        CAUTION: the pointcloud and the last received time in the odometry may be very close. As a result,
        the function get_closest_pose_at_time must be used instead of the function interpolated_pose_at_time
        get_closest_pose_at_time: the last received odometry, closest to the current time.
        interpolated_pose_at_time: needs two times
        """
        # # current timestamp
        timestamp = msg.header.stamp.to_sec()
        Tg = self.perform_local_scanmatching(msg)
        # publish last
        if Tg is not None:
            self.publish_pose(Tg, timestamp=timestamp)
        return

    def publish_pose(self, T, timestamp):
        """
        Publishing the global transform of the scanmatching at that time.
        """
        if T is None:
            return
        # print('Publishing last pose:')
        position = T.pos()
        orientation = T.Q()
        msg = Odometry()
        # Yes! caution. Publishing scanmatching transformation at the corresponding timestamp (in
        # the past)
        msg.header.stamp = rospy.Time.from_sec(timestamp)
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


    def timer_callback_plot_info(self, event):
        # print(50 * '*')
        # print('Number of pointclouds: ')
        # print('len: ', len(self.pcdbuffer))
        # print(50 * '*')

        # odom_times = np.array(self.times_odometry) - self.start_time
        # lidar_times = np.array(self.times_lidar) - self.start_time
        # ax.clear()
        # if len(odom_times) > 1:
        #     ax.scatter(odom_times, np.ones(len(odom_times)), marker='.', color='blue')
        # if len(lidar_times) > 1:
        #     ax.scatter(lidar_times, np.ones(len(lidar_times)), marker='.', color='red')

        # PLOT LOCALIZATION
        print('Odombuffer length: ', len(self.odombuffer.times))
        # print('LidarBuffer length: ', len(self.pcdbuffer.times))
        odo_positions = self.odombuffer.get_positions()
        positions_sm = []
        for i in range(len(self.global_transforms)):
            T = self.global_transforms[i][0]
            positions_sm.append(T.pos())
        positions_sm = np.array(positions_sm)

        ax1.clear()
        if len(odo_positions) > 0:
            ax1.scatter(odo_positions[:, 0], odo_positions[:, 1], marker='.', color='red', label='Odometry')

        if len(positions_sm) > 0:
            # positions_sm = np.array(self.positions_sm)
            ax1.scatter(positions_sm[:, 0], positions_sm[:, 1], marker='.', color='blue', label='Scanmatcher')
        # if len(self.utm_valid_positions) > 0:
        #     utm_valid_positions = np.array(self.utm_valid_positions)
        #     ax.scatter(utm_valid_positions[:, 0],
        #                utm_valid_positions[:, 1], marker='.', color='red')
        canvas1.print_figure('plots/run_scanmatcher_plot1.png', bbox_inches='tight')

        ax2.clear()
        frequency = np.array(self.frequency)
        if len(frequency) > 0:
            ax2.plot(frequency, marker='.', color='red', label='Frequency (Hz)')
        canvas2.print_figure('plots/run_scanmatcher_plot2.png', bbox_inches='tight')

    def perform_local_scanmatching(self, msg):
        start = time.time()
        d_poses = PARAMETERS.config.get('scanmatcher').get('d_poses')
        th_poses = PARAMETERS.config.get('scanmatcher').get('th_poses')
        voxel_size = PARAMETERS.config.get('scanmatcher').get('voxel_size')
        voxel_size_normals = PARAMETERS.config.get('scanmatcher').get('normals').get('voxel_size_normals')
        max_nn_normals = PARAMETERS.config.get('scanmatcher').get('normals').get('max_nn_normals')
        # current timestamp
        timestamp = msg.header.stamp.to_sec()

        self.time_diffs.append(timestamp - self.odombuffer.times[-1])

        if len(self.odombuffer.times) == 0:
            print('Received pointcloud but no odometry yet. Waiting for odometry')
            return

        if self.pcd1 is None:
            # odo_ti, _ = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp,
            #                                                       delta_threshold_s=delta_threshold_s)
            # WORKING odo_ti, _ = self.odombuffer.get_closest_pose_at_time(timestamp=timestamp,
            #                                                     delta_threshold_s=delta_threshold_s)
            odo_ti, _, _ = self.odombuffer.interpolated_pose_at_time_new(timestamp=timestamp)
            # this may happen only when the system is startting and no odometry readings exist
            if odo_ti is None:
                print('No corresponding odo_ti found, skipping')
                print(10 * "odo_tI error")
                # time.sleep(self.time_sleep_synchro)
                # self.time_sleep_synchro += 0.5
                return
            pcd = LidarScan(time=timestamp, pose=odo_ti)
            pcd.load_pointcloud_from_msg(msg=msg)
            self.pcd1 = pcd
            self.pcd1.down_sample(voxel_size=voxel_size)
            self.pcd1.filter_points()
            self.pcd1.estimate_normals(voxel_size_normals=voxel_size_normals,
                                       max_nn_estimate_normals=max_nn_normals)
            T0 = HomogeneousMatrix()
            # adding global transform and pcd
            self.global_transforms.append((T0, timestamp))
            # self.publish_pose(T0, timestamp=timestamp)
            return T0

        # read the newly received pointcloud odometry
        # odo_tj, _ = self.odombuffer.get_closest_pose_at_time(timestamp=timestamp, delta_threshold_s=delta_threshold_s)
        # odo_tj, _ = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp,
        #                                                       delta_threshold_s=delta_threshold_s)
        odo_tj, _, _ = self.odombuffer.interpolated_pose_at_time_new(timestamp=timestamp)
        if odo_tj is None:
            print('Caution: no interpolated odometry odo_tj found')
            print(10*"odo_tJ error")
            # time.sleep(self.time_sleep_synchro)
            # self.time_sleep_synchro+=0.5
            return None

        pcd = LidarScan(time=timestamp, pose=odo_tj)
        pcd.load_pointcloud_from_msg(msg=msg)
        # the last received p
        self.pcd2 = pcd
        # compute initial transform from odometry
        odoi = self.pcd1.pose.T()
        odoj = self.pcd2.pose.T()
        Tij0 = odoi.inv() * odoj

        d = np.linalg.norm(Tij0.pos())
        th = min(Tij0.R().euler()[0].abg[0],Tij0.R().euler()[1].abg[0])
        print('d, th: ', d, th)
        if d < d_poses and th < th_poses:
            print('Caution: no movement found')
            print("No movement")
            return None

        # process pcd2
        self.pcd2.down_sample(voxel_size=voxel_size)
        self.pcd2.filter_points()
        self.pcd2.estimate_normals(voxel_size_normals=voxel_size_normals,
                                   max_nn_estimate_normals=max_nn_normals)

        Tij = self.scanmatcher.registration(self.pcd1, self.pcd2, Tij_0=Tij0)
        # draw registration result
        # self.relative_transforms.append((Tij, timestamp))
        # append to global transforms
        # Ti = self.global_transforms[-1][0]
        Ti = self.Tg
        Tg = Ti * Tij
        self.Tg = Tg
        # adding global transform and pcd
        self.global_transforms.append((Tg, timestamp))

        # swap pointclouds
        self.pcd1 = self.pcd2
        end = time.time()
        # self.computation_times.append(end-start)
        self.frequency.append(1 / (end - start))
        return Tg

if __name__ == "__main__":
    node = ScanmatchingNode()
    node.run()


