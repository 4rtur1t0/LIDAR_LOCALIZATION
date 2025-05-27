"""
A simple Scanmatcher for LiDAR pointclouds.
Publishes a global estimated pose based on the first odometry reading found.
No GPS or IMU is integrated.
Uses Open3d for ICP

"""
import rospy
import numpy as np
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
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)
        t1 = PARAMETERS.config.get('scanmatcher').get('threads').get('seconds_period_scanmatching_thread')
        t2 = PARAMETERS.config.get('scanmatcher').get('threads').get('seconds_period_publish_thread')
        t3 = PARAMETERS.config.get('scanmatcher').get('threads').get('seconds_period_plot_info_thread')
        # on a different timed thread, process pcds
        rospy.Timer(rospy.Duration(secs=0, nsecs=int(t1*1e9)), self.timer_callback_process_scanmatching)
        # on a different timed thread, publish the found transforms
        rospy.Timer(rospy.Duration(secs=0, nsecs=int(t2*1e9)), self.timer_callback_publish_transforms)
        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(secs=0, nsecs=int(t3*1e9)), self.timer_callback_plot_info)

        # Publisher
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)
        # stores odometry poses as a short buffer with deque
        self.odombuffer = PosesBuffer(maxlen=1000)
        # the lidar buffer
        self.pcdbuffer = LidarBuffer(maxlen=30)
        # store the results from the beginning of the experiment
        self.relative_transforms = []
        self.global_transforms = []
        self.last_global_transform_published = 0
        self.start_time = None


        self.times_odometry = []
        self.times_lidar = []
        self.positions_sm = []
        # the scanmatching object
        self.scanmatcher = ScanMatcher()
        self.timer_callback_process_scanmatching_computation_time = []
        rospy.loginfo("ScanMatcher with odom/pc running.")

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
        Get last pcd reading and append to buffer.
        To save memory, pointclouds are appended if enough distance/angle is traversed (in odometry)
        """
        timestamp = msg.header.stamp.to_sec()
        # if self.start_time is None:
        #     self.start_time = timestamp
        self.times_lidar.append(timestamp)

        if len(self.odombuffer.times) == 0:
            print('Received pointcloud but no odometry yet. Waiting for odometry')
            return
        print('Received pointcloud')
        # add first pointcloud in this particular case:  only try to get the first pointcloud
        # if we have at least one odometry meeasurement and no initial pcd has been included
        # in order to find a proper initial Tij_0
        if len(self.pcdbuffer.times) == 0: # and (len(self.odombuffer.times) > 0):
            print(30*'+')
            print('add_first_pcd')
            # CAUTION!!! for some reason, the pointclouds and the odometry come at different times
            # IN PARTICULAR, ODOMETRY COMES LATER IN TIME.
            # wait 0.1 s to allow odometry buffer to fill up (approximately)
            #delay = PARAMETERS.config.get('scanmatcher').get('initial_transform').get('delay_seconds_lidar_odometry')
            #rospy.sleep(delay)
            self.add_first_pcd(timestamp=timestamp, msg=msg)
            return
        delta_threshold_s = PARAMETERS.config.get('scanmatcher').get('initial_transform').get('delta_threshold_s')
        odo_ti = self.pcdbuffer[-1].pose
        odo_tj, _ = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp, delta_threshold_s=delta_threshold_s)
        if odo_tj is None:
            print('ERROR: !NO VALID ODOMETRY FOUND AT TIMESTAMP: ')
            print(1000*'!')
            return
        # Now, only add another pointcloud if the odometry is significantly moved
        d, th = compute_rel_distance(odo_ti, odo_tj)
        # if the distance is larger or the angle is larger that... add pcd to buffer
        d_poses = PARAMETERS.config.get('scanmatcher').get('d_poses')
        th_poses = PARAMETERS.config.get('scanmatcher').get('th_poses')
        if d > d_poses or th > th_poses:
            print(50*'=')
            # print('Adding new pointcloud at: ', d, ', ', th)
            print('Found lidar nicely separated in odometry')
            print('Appending pointcloud')
            print(50*'=')
            pcd = LidarScan(time=timestamp, pose=odo_tj)
            pcd.load_pointcloud_from_msg(msg=msg)
            self.pcdbuffer.append(pcd=pcd, time=timestamp)
            print(30 * '+')
        return

    def timer_callback_process_scanmatching(self, event):
        """
        Given pcd1, tpcd1 --> find interpolated odom at tpcd1
        Given pcd2, tpcd2 --> find interpolated odom at tpcd1
        Compute Tij0 from interpolated odom.
        Compute registration
        """
        start_time = time.time()
        print('timer_callback_process_scanmatching!!')
        print('CURRENT POINTCLOUD BUFFER used percentage is: ', 100*len(self.pcdbuffer.times)/self.pcdbuffer.times.maxlen)
        print(30*'=')
        voxel_size = PARAMETERS.config.get('scanmatcher').get('voxel_size')
        # compute the scanmatching on the clouds stored at the pcdbuffer
        k = 0
        # fix the number of pointclouds to process
        N = len(self.pcdbuffer)
        # preprocess pointclouds first
        for i in range(N):
            self.pcdbuffer[i].down_sample(voxel_size=voxel_size)
            self.pcdbuffer[i].filter_points()
            self.pcdbuffer[i].estimate_normals()
        # Now iterate and compute
        for i in range(N-1):
            print('Tiempo lidar', self.pcdbuffer.times[i] - self.start_time)
            # self.pcdbuffer[i].draw_cloud()
            odoi = self.pcdbuffer[i].pose.T()
            odoj = self.pcdbuffer[i+1].pose.T()
            Tij0 = odoi.inv()*odoj
            # self.pcdbuffer[i].down_sample(voxel_size=voxel_size)
            # self.pcdbuffer[i + 1].down_sample(voxel_size=voxel_size)
            # self.pcdbuffer[i].filter_points()
            # self.pcdbuffer[i+1].filter_points()
            # self.pcdbuffer[i].estimate_normals()
            # self.pcdbuffer[i + 1].estimate_normals()
            Tij = self.scanmatcher.registration(self.pcdbuffer[i], self.pcdbuffer[i+1], Tij_0=Tij0)
            # draw registration result
            self.relative_transforms.append((Tij, self.pcdbuffer.times[i+1]))
            # append to global transforms
            Ti = self.global_transforms[-1][0]
            Tg = Ti*Tij
            # adding global transform and pcd
            self.global_transforms.append((Tg, self.pcdbuffer.times[i+1]))
            self.positions_sm.append(Tg.pos())
            k += 1
        # deque pointclouds and free memory
        for j in range(k):
            print('Deque used pointclouds')
            self.pcdbuffer.popleft()
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.timer_callback_process_scanmatching_computation_time.append(elapsed_time)
        print(30*'*')
        print(f"timer_callback_process_scanmatching Execution time: {elapsed_time:.3f} seconds")
        print(30 * '*')

    def timer_callback_publish_transforms(self, event):
        """
        Publish transforms at its given times
        """
        print('Publish transforms!!')
        print(30 * '=')
        if len(self.global_transforms) == 0:
            return
        # compute global transforms
        # Tg = self.global_transforms[0][0]
        # self.positions_sm = []
        # self.positions_sm.append(Tg.pos())
        # for i in range(len(self.relative_transforms)):
        #     Tij = self.relative_transforms[i][0]
        #     timestamp = self.relative_transforms[i][1]
        #     Tg = Tg*Tij
        #     self.global_transforms.append((Tg, timestamp))
        #     self.positions_sm.append(Tg.pos())
        first_index = self.last_global_transform_published
        for i in range(first_index, len(self.global_transforms)):
            # if i > self.last_global_transform_published:
            self.publish_pose(T=self.global_transforms[i][0], timestamp=self.global_transforms[i][1])
            self.last_global_transform_published += 1

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

    def add_first_pcd(self, timestamp, msg):
        """
        Caution: this covers the case in which a LiDAR pcd has been received and:
        a) Only one has been found--> the closest time is annexed.
        b) Or two times of odometry are found -->
        """
        # caution, considering twice this time, for the first pointcloud
        # this is the max. time to issue a warning!
        delta_threshold_s = PARAMETERS.config.get('scanmatcher').get('initial_transform').get('delta_threshold_s')
        print('Received first pointcloud with odometry')
        if len(self.odombuffer.times) == 0:
            print('NO ODO MEASUREMENTS FOUND. WAIT FOR ODOMETRY')
            return
        # first try to get an interpolated pose
        odo_t0 = None
        try:
            print('Getting interp odo. Two odometry measurements exist.')
            odo_t0, _ = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp,
                                                                delta_threshold_s=2*delta_threshold_s)
        except:
            pass
        # in any case, try to get the closest pose
        if odo_t0 is None:
            try:
                print('Getting closets odo. Only one odometry measurement exists.')
                odo_t0, _ = self.odombuffer.get_closest_pose_at_time(timestamp=timestamp,
                                                              delta_threshold_s=2*delta_threshold_s)
            except:
                 pass
        if odo_t0 is None:
            print('Skipping pointcloud. No close odom found. Wait for next pointcloud')
            return None
        # if odometry has been found in any case,
        pcd = LidarScan(time=timestamp, pose=odo_t0)
        pcd.load_pointcloud_from_msg(msg=msg)
        self.pcdbuffer.append(pcd=pcd, time=timestamp)
        # convert to transform and store
        T0 = odo_t0.T()
        # first global transform and position
        self.global_transforms = [(T0, timestamp)]
        self.positions_sm = [T0.pos()]
        # no transforms published so far
        self.last_global_transform_published = -1
        return odo_t0

    def timer_callback_plot_info(self, event):
        print(50 * '*')
        print('Number of pointclouds: ')
        print('len: ', len(self.pcdbuffer))
        print(50 * '*')

        # odom_times = np.array(self.times_odometry) - self.start_time
        # lidar_times = np.array(self.times_lidar) - self.start_time
        # ax.clear()
        # if len(odom_times) > 1:
        #     ax.scatter(odom_times, np.ones(len(odom_times)), marker='.', color='blue')
        # if len(lidar_times) > 1:
        #     ax.scatter(lidar_times, np.ones(len(lidar_times)), marker='.', color='red')

        # PLOT LOCALIZATION
        print('Odombuffer length: ', len(self.odombuffer.times))
        print('LidarBuffer length: ', len(self.pcdbuffer.times))
        odo_positions = self.odombuffer.get_positions()

        ax1.clear()
        if len(odo_positions) > 0:
            ax1.scatter(odo_positions[:, 0], odo_positions[:, 1], marker='.', color='red', label='Odometry')

        if len(self.positions_sm) > 0:
            positions_sm = np.array(self.positions_sm)
            ax1.scatter(positions_sm[:, 0], positions_sm[:, 1], marker='.', color='blue', label='Scanmatcher')
        # if len(self.utm_valid_positions) > 0:
        #     utm_valid_positions = np.array(self.utm_valid_positions)
        #     ax.scatter(utm_valid_positions[:, 0],
        #                utm_valid_positions[:, 1], marker='.', color='red')
        canvas1.print_figure('plots/scanmatcher_plot1.png', bbox_inches='tight')

        ax2.clear()
        computation_times = np.array(self.timer_callback_process_scanmatching_computation_time)
        if len(computation_times) > 0:
            ax2.plot(computation_times, marker='.', color='red', label='Computation times (s)')
        canvas2.print_figure('plots/scanmatcher_plot2.png', bbox_inches='tight')


if __name__ == "__main__":
    node = ScanmatchingNode()
    node.run()


