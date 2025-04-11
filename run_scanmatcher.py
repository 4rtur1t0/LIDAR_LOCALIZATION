"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from graphslam.graphSLAM import GraphSLAM
from artelib.homogeneousmatrix import HomogeneousMatrix
from observations.lidarbuffer import LidarBuffer, LidarScan
from observations.posesarray import Pose
from config import PARAMETERS
from observations.posesbuffer import PosesBuffer
from tools.gpsconversions import gps2utm
from sensor_msgs.msg import PointCloud2
from scanmatcher.scanmatcher import ScanMatcher

fig, ax = plt.subplots()
canvas = FigureCanvas(fig)

ODOMETRY_TOPIC = '/husky_velocity_controller/odom'
POINTCLOUD_TOPIC = '/ouster/points_low_rate'
OUTPUT_TOPIC = '/odometry_lidar_scanmatching'


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
        print('Initializing scanmatching node!')
        rospy.init_node('scan_matching_sync_node')
        print('Subscribing to ODOMETRY and pointclouds')
        print('CAUTION: odometry and poinclouds are synchronized with filter messages')
        print('WAITING FOR MESSAGES!')
        # Subscriptions
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)

        # on a different timed thread, process pcds
        # rospy.Timer(rospy.Duration(secs=0, nsecs=int(1*1e9)), self.timer_callback_process_scanmatching)
        # on a different timed thread, publish the found transforms
        # rospy.Timer(rospy.Duration(secs=0, nsecs=int(1*1e9)), self.timer_callback_publish_transforms)
        # Set up a timer to periodically update the plot
        # rospy.Timer(rospy.Duration(secs=1), self.timer_callback_plot_info)

        # Publisher
        self.pub = rospy.Publisher(OUTPUT_TOPIC, Odometry, queue_size=10)
        # stores odometry poses as a short buffer with deque
        self.odombuffer = PosesBuffer(maxlen=1000)
        self.pcdbuffer = LidarBuffer(maxlen=100)
        # store the results from the beginning of the experiment
        self.relative_transforms = []
        self.global_transforms = []
        self.last_global_transform_published = -1
        self.start_time = None

        # init global transforms
        # T0 = HomogeneousMatrix()
        # self.global_transforms.append(T0)
        # stores the last time published
        # self.global_transforms_times = [0]
        self.scanmatcher = ScanMatcher()
        rospy.loginfo("Scan Matcher with odom/pc running.")
        # rospy.spin() !!

    def run(self):
        rospy.spin()

    def odom_callback(self, msg):
        """
        Get last odom reading and append to buffer.
        """
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
        pose = Pose()
        pose.from_message(msg.pose.pose)
        timestamp = msg.header.stamp.to_sec()
        self.odombuffer.append(pose, timestamp)

    def pc_callback(self, msg):
        """
        Get last pcd reading and append to buffer.
        To save memory, pointclouds are appended if enough distance/angle is traversed (in odometry)
        """
        if self.start_time is None:
            self.start_time = msg.header.stamp.to_sec()
        print('Received pointcloud')
        timestamp = msg.header.stamp.to_sec()
        # add first pointcloud
        #if len(self.pcdbuffer.times) == 0:
        # only try to get the first pointcloud if we have at least one odometry meeasurement
        if len(self.odombuffer.times) == 0:
            odo_t0 = self.add_first_pcd(timestamp=timestamp, msg=msg)
            T0 = odo_t0.T()
            self.global_transforms = [(T0, timestamp)]
            # no transforms published so far
            self.last_global_transform_published = -1
            return
        odo_ti = self.pcdbuffer[-1].pose
        odo_tj = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp)
        if odo_tj is None:
            print('NO VALID ODOMETRY FOUND AT TIMESTAMP: ')
            return
        d, th = compute_rel_distance(odo_ti, odo_tj)
        # if the distance is larger or the angle is larger thatn... add pcd to buffer
        if d > 0.2 or th > 0.1:
            print(50*'=')
            print('Adding new pointcloud at: ', d, ', ', th)
            print('Found lidar nicely separated in odometry')
            print('Appending pointcloud')
            print(50*'=')
            pcd = LidarScan(time=timestamp, pose=odo_tj)
            pcd.load_pointcloud_from_msg(msg=msg)
            self.pcdbuffer.append(pcd=pcd, time=timestamp)
            print(30 * '+')
        return

    # df gps_callback(self, ms)
    def timer_callback_process_scanmatching(self, event):
        """
        Given pcd1, tpcd1 --> find interpolated odom at tpcd1
        Given pcd2, tpcd2 --> find interpolated odom at tpcd1
        Compute Tij0 from interpolated odom.
        Compute registration
        """
        print('timer_callback_process_scanmatching!!')
        print(30*'=')
        i = 0
        for i in range(len(self.pcdbuffer)-1):
            print('Tiempo lidar', self.pcdbuffer.times[i] - self.start_time)
            # self.pcdbuffer[i].draw_cloud()
            odoi = self.pcdbuffer[i].pose.T()
            odoj = self.pcdbuffer[i+1].pose.T()
            Tij0 = odoi.inv()*odoj
            self.pcdbuffer[i].filter_points()
            self.pcdbuffer[i+1].filter_points()
            self.pcdbuffer[i].estimate_normals()
            self.pcdbuffer[i + 1].estimate_normals()
            Tij = self.scanmatcher.registration(self.pcdbuffer[i], self.pcdbuffer[i+1], Tij_0=Tij0)
            self.relative_transforms.append(Tij)
            Ti = self.global_transforms[-1][0]
            # adding global transform and pcd
            self.global_transforms.append((Ti*Tij, self.pcdbuffer.times[i+1]))

        # print('Deque used pointclouds')
        # for j in range(i+1):
        #     self.pcdbuffer.popleft()

    def timer_callback_publish_transforms(self, event):
        """
        Publish transforms at its given times
        """
        print('Publish transforms!!')
        print(30 * '=')
        for i in range(len(self.global_transforms)):
            if i > self.last_global_transform_published:
                self.publish_pose(T=self.global_transforms[i][0], timestamp=self.global_transforms[i][1])
                self.last_global_transform_published = i

    def timer_callback_plot_info(self, event):
        # print(Odombuffer'')
        print('Odombuffer length: ', len(self.odombuffer.times))
        print('LidarBuffer length: ', len(self.pcdbuffer.times))
        positions = self.odombuffer.get_positions()
        # positions = self.graphslam.get_solution_positions()
        ax.clear()
        if len(positions) > 0:
            ax.scatter(positions[:, 0], positions[:, 1], marker='.', color='blue')
        # if len(self.utm_valid_positions) > 0:
        #     utm_valid_positions = np.array(self.utm_valid_positions)
        #     ax.scatter(utm_valid_positions[:, 0],
        #                utm_valid_positions[:, 1], marker='.', color='red')
        canvas.print_figure('plot.png', bbox_inches='tight')

    def publish_pose(self, T, timestamp):
        """
        Publishing the global transform of the scanmatching at that time.
        """
        if T is None:
            return
        print('Publishing last pose:')
        position = T.pos()
        orientation = T.Q()
        msg = Odometry()
        msg.header.stamp = timestamp
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
        print('Received first pointcloud with odometry')
        #if len(self.odombuffer.times) == 0:
        #    print('NO ODO MEASUREMENTS FOUND. WAIT FOR ODOMETRY')
        #    return
        if len(self.odombuffer.times) == 1:
            print('Getting closets odo. Only one odometry measurement exists.')
            odo_t1 = self.odombuffer.closest_pose_at_time(timestamp=timestamp)
        else:
            print('Getting interp odo. Two odometry measurements exist.')
            odo_t1 = self.odombuffer.interpolated_pose_at_time(timestamp=timestamp)
        if odo_t1 is None:
            print('Skipping pointcloud. No close odom found. Wait for next pointcloud')
            return
        pcd = LidarScan(time=timestamp, pose=odo_t1)
        pcd.load_pointcloud_from_msg(msg=msg)
        self.pcdbuffer.append(pcd=pcd, time=timestamp)
        return odo_t1



    # def pc_callback(self, msg):
    #     timestamp = msg.header.stamp.to_sec()
    #     matched_odom = self.find_closest_odom(timestamp)
    #
    #     if self.start_time is None:
    #         self.start_time = timestamp
    #     if matched_odom is None:
    #         rospy.logwarn("No odometry available for this scan.")
    #         return
    #     if self.posei is None: # Convert odometry to matrix
    #         posei = Pose()
    #         posei.from_message(matched_odom)
    #         self.posei = posei.T()
    #         rospy.loginfo("Initialized first odometry.")
    #         return
    #     print('Current experiment time is: ', timestamp - self.start_time)
    #     # the current new pose
    #     posej = Pose()
    #     posej.from_message(matched_odom)
    #     self.posej = posej.T()
    #     Tij = self.posei.inv() * self.posej
    #     # toggle poses
    #     self.posei = posej
    #
    #     # filter by distance and angle
    #     d = np.linalg.norm(Tij.pos())
    #
    #     if self.pcd_i is None:
    #         self.pcd_i = LiDARScan()
    #         self.pcd_i.load_pointcloud_from_msg(msg)
    #         rospy.loginfo("Initialized first scan.")
    #         return
    #
    #
    #
    #     cloud = np.array([p[:3] for p in pc2.read_points(msg, skip_nans=True)])
    #     source_pcd = o3d.geometry.PointCloud()
    #     source_pcd.points = o3d.utility.Vector3dVector(cloud)
    #
    #
    #
    #     delta_odom = np.linalg.inv(self.prev_odom_matrix) @ current_odom_matrix
    #
    #     reg = o3d.pipelines.registration.registration_icp(
    #         source_pcd, self.prev_pcd, 1.0, delta_odom,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #     )
    #
    #     T = reg.transformation
    #     rospy.loginfo(f"ICP fitness: {reg.fitness:.3f}")
    #
    #     self.publish_pose(T, msg.header.stamp, msg.header.frame_id)
    #
    #     self.prev_pcd = source_pcd
    #     self.prev_odom_matrix = current_odom_matrix
    #
    # def synced_callback(self, pc_msg, odom_msg):
    #     timestamp = pc_msg.header.stamp.to_sec()
    #     if self.posei is None: # Convert odometry to matrix
    #         posei = Pose()
    #         posei.from_message(odom_msg.pose.pose)
    #         self.posei = posei.T()
    #         self.start_time = timestamp
    #         return
    #     print('Current experiment time is: ', timestamp - self.start_time)
    #     posej = Pose()
    #     posej.from_message(odom_msg.pose.pose)
    #     self.posej = posej.T()
    #     Tij = self.posei.inv()*self.posej
    #     # check for relative distance traveled
    #
    #
    #     # Convert PointCloud2 to Open3D
    #     cloud = np.array([p[:3] for p in pc2.read_points(pc_msg, skip_nans=True)])
    #     source_pcd = o3d.geometry.PointCloud()
    #     source_pcd.points = o3d.utility.Vector3dVector(cloud)
    #
    #     if self.prev_pcd is None or self.prev_odom_matrix is None:
    #         self.prev_pcd = source_pcd
    #         self.prev_odom_matrix = current_odom_matrix
    #         rospy.loginfo("Stored first scan & odometry.")
    #         return
    #
    #     # Relative odometry as initial guess
    #     delta_odom = np.linalg.inv(self.prev_odom_matrix) @ current_odom_matrix
    #
    #     reg = o3d.pipelines.registration.registration_icp(
    #         source_pcd, self.prev_pcd, 1.0, delta_odom,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #     )
    #
    #     T = reg.transformation
    #     rospy.loginfo(f"ICP fitness: {reg.fitness:.3f}")
    #
    #     self.publish_pose(T, pc_msg.header.stamp, pc_msg.header.frame_id)
    #
    #     self.prev_pcd = source_pcd
    #     self.prev_odom_matrix = current_odom_matrix
    #
    #
    # def odom_callback(self, msg):
    #     print('Received odo measurement')
    #     timestamp = msg.header.stamp.to_sec()
    #     print('Current experiment time is: ', timestamp-self.start_time)
    #     if self.posei is None:
    #         posei = Pose()
    #         posei.from_message(msg.pose.pose)
    #         self.posei = posei
    #         self.start_time = timestamp
    #         return
    #     posej = Pose()
    #     posej.from_message(msg.pose.pose)
    #     self.posej = posej
    #     Ti = self.posei.T()
    #     Tj = self.posej.T()
    #     Tij = Ti.inv()*Tj
    #
    #     self.publish_pose(last_sol)

    # def gps_callback(self, msg):
    #     # Convert lat/lon to dummy XYZ — replace with ENU or UTM in real use
    #     print(30*'$')
    #     print('Received GPS reading')
    #     print(msg.status)
    #     print('Sigma is: ',
    #
    #           np.sqrt(msg.position_covariance[0]))
    #     print(30 * '$')
    #     self.last_gps = convert_and_filter_gps(msg)
    #     if self.last_gps is None:
    #         print(30 * '?')
    #         print("Received non-valid GPS reading!")
    #         print(30 * '?')
    #         return
    #     print(30*'*')
    #     print("Received valid GPS reading!!")
    #     print(30 * '*')






if __name__ == "__main__":
    node = ScanmatchingNode()
    node.run()


