"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.


    This node subscribes to the localized_pose topic.
    The localized_pose topic is initially published by the localization node.
    The initial pose is used to find a number of close pointclouds in the map. A registration is then performed
    As a result, we end up having another prior3Dfactor observation on the state X(i)

"""
import rospy
from nav_msgs.msg import Odometry
from observations.lidarbuffer import LidarBuffer, LidarScan
from observations.posesbuffer import PosesBuffer, Pose
from sensor_msgs.msg import PointCloud2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from observations.posesbuffer import PosesBuffer
import time
import open3d as o3d
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)

MAP_DIRECTORY = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
DELTA_THRESHOLD_S = 1.0 # for interpolation on localized_pose
# CAUTION: this topic must be subscribed to the /ouster/points (high rate) topic
POINTCLOUD_TOPIC = '/ouster/points_low_rate'
# POINTCLOUD_TOPIC = '/ouster/points'

# INITIAL ESTIMATION POSE, this is the output of the run_graph_localizer algorithm
INITIAL_ESTIMATED_POSE = '/localized_pose'

# WE ARE PUBLISHING THE PRIOR ESTIMATION HERE, as a result of the localization in the map
MAP_SM_GLOBAL_POSE_TOPIC = '/map_sm_global_pose'


class GlobalMap():
    def __init__(self, map_directory, map_filename):
        # 1. Load the global map (PCD)
        map_filename = map_directory + '/' + map_filename #'/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17/'
        self.global_map = o3d.io.read_point_cloud(map_filename)
        print("[INFO] Global map loaded with", len(self.global_map.points), "points.")
        # the path ground trutth (FOR THE MAP)
        # self.ground_truth_path = self.read_data_tum(directory=map_directory)
        # 3. Downsample for faster processing the global map
        voxel_size_global_map = 0.2  # adjust as needed
        self.global_map.voxel_down_sample(voxel_size_global_map)
        # 4. Estimate normals (required for registration)
        self.global_map.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3,
                                                                     max_nn=50))


    def perform_global_localization_of_scan_i(self, local_scan, initial_guess, show=False):
        """
        # the local scan is registered agaistnthe global map.
        It is important to start with a nice estimation
        In addition, the min and max radius of th elocal scan must not be very large.
        These radius have to be in accordance with the filtered submap (currently +-20m centerede
        at the current robot position)
        """
        voxel_size_local_scan = 0.2
        # show = True
        print("[INFO] Local scan loaded with", len(local_scan.pointcloud.points), "points.")

        # local_scan.filter_height(heights=heights)
        local_scan.down_sample(voxel_size=voxel_size_local_scan)
        local_scan.filter_radius(radii=[0.5, 12.0])
        local_scan.estimate_normals(voxel_size_normals=voxel_size_local_scan, max_nn_estimate_normals=50)
        print("[INFO] Local scan filtered, normals and down_sampled at", len(local_scan.pointcloud.points), "points.")
        # 2. Load or simulate a new LiDAR scan (local point cloud)
        # local_scan = o3d.io.read_point_cloud("local_scan.pcd")

        # reduce to a local map, reduce the number of points
        # filter a submap of 20x20 centered on the current xy position
        global_map_temp = self.global_map
        x = initial_guess.array[0][3]
        y = initial_guess.array[1][3]
        ##
        # self.global_map_temp.filter_coordinates(x_limits=(x-12, x+12), y_limits=(y-12,y+12), z_limits=(-20, 20))
        x_min = x-20.0
        x_max = x+20
        y_min = y-20
        y_max = y+20
        z_min = -20
        z_max = 20
        points = np.asarray(global_map_temp.points)
        normals = np.asarray(global_map_temp.normals)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        idx = np.where((x > x_min) & (x < x_max) & (y > y_min) & (y < y_max) & (z > z_min) & (z < z_max))
        global_map_temp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx]))
        # Do not compute normals: just copy them as filtered.
        global_map_temp.normals = o3d.utility.Vector3dVector(normals[idx])
        ##

        # 6. ICP registration (use as NDT substitute)
        reg_result = o3d.pipelines.registration.registration_icp(
            local_scan.pointcloud, global_map_temp, max_correspondence_distance=8.0,
            init=initial_guess.array,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        print("[RESULT] Transformation matrix:\n", reg_result.transformation)
        print("[INFO] Fitness:", reg_result.fitness, "Inlier RMSE:", reg_result.inlier_rmse)

        if show:
            # 7. Visualize result
            local_scan.transform(reg_result.transformation)
            # filter height of result
            points = np.asarray(global_map_temp.points)
            # [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            idx2 = np.where((points[:, 2] > -1.0) & (points[:, 2] < 1.5))
            global_map_filtered = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))
            global_map_filtered.paint_uniform_color([1.0, 0, 0])
            local_scan.pointcloud.paint_uniform_color([0, 0, 1.0])
            # o3d.visualization.draw_geometries([local_scan.pointcloud])
            o3d.visualization.draw_geometries([global_map_filtered, local_scan.pointcloud])
        return HomogeneousMatrix(reg_result.transformation)


class GlobalScanMatchingROSNode:
    def __init__(self):
        self.start_time = None
        # store odometry in deque fashion
        self.localized_pose_buffer = PosesBuffer(maxlen=5000)

        # LOAD THE MAP
        print('Loading MAP')
        self.global_map = GlobalMap(map_directory=MAP_DIRECTORY, map_filename='global_map.pcd')
        print('Map loaded!')

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        rospy.init_node('scanmatcher_to_map_node')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')

        # Subscriptions to the pointcloud topic and to the
        # CAution!!! There must be a delay between the received pointcloud and the
        # /localized_pose trajectory. In particular, the pointcloud times must be
        # retarded with respect to the /localized_pose
        # Caution: queue_size must be 1 or 2 approxx
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback, queue_size=10)
        # subscription to the current localized pose
        rospy.Subscriber(INITIAL_ESTIMATED_POSE, Odometry, self.localized_pose_callback)

        # Set up a timer to periodically update the graph
        rospy.Timer(rospy.Duration(5), self.plot_timer_callback)
        # Publisher
        self.pub = rospy.Publisher(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, queue_size=10)

        # print info.
        self.all_initial_estimations = [] # initial without map
        self.all_refined_estimations = [] # compared to map
        # caution: skipping some of the pointclouds
        self.times_pcd = []
        self.times_pose_buffer = []
        self.diff_times_pcd_pose = []

        self.lidar_array = deque()

    def pc_callback(self, msg):
        """
        Get last pcd reading and try to localize it
        """
        start_time = time.time()
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        print('Received pointcloud')
        try:
            pcd = LidarScan(time=timestamp, pose=None)
            pcd.load_pointcloud_from_msg(msg=msg)

            # given the current /localized_pose, refine this information by
            # registering the current scan against the global map.
            self.compute_global_scanmatching(pcd, timestamp_pcd=timestamp)
            end_time = time.time()
            print(30 * '+')
            print('Received and loaded pointcloud')
            print(f"pc_callback time:, {end_time - start_time:.4f} seconds")
            print(30 * '+')
        except:
            print('Exception captured in compute_global_scanmatching!!!')
            pass
        return

    def localized_pose_callback(self, msg):
        """
            Obtain the last estimations on the robot path
            This /localized_pose is used as an initial estimation and usually obtained from the localization node itself.
            This should be the /localized_pose topic, which maintains
            the last localization with all the information, excluding
            the localization with respec to the map.
        """
        print(50*'_')
        print('Received localized pose!!!')
        print('localized_pose_buffer size: ', len(self.localized_pose_buffer))
        print(50 * '_')
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.localized_pose_buffer.append(pose, timestamp)

    def compute_global_scanmatching(self, pcd, timestamp_pcd):
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

            # get the state X(i), call it T0i (obtain the closest pcd in time)
            # get the closest pointcloud L(j), call it T0j. Here we use the closest L(j) in Euclidean
            # compute an initial estimation Tij. # it must be: T0i*Tij = T0j --> thus Tij = T0i.inv()*T0j
            # given the initial transformation Tij, compute Tij_ using ICP
            # it must be: T0i_*Tij_ = T0j --> thus T0i_ = T0j*Tij_.inv(), This last T0i_ is then published and
            # must be added as a prior
            CAUTION: the process
        """
        start = time.time()
        if len(self.localized_pose_buffer) == 0:
            print("\033[91mCaution!!! No initial /localized_pose received yet.\033[0m")
            return
        # difference, pcd last localized pose
        # diff = timestamp_pcd - self.localized_pose_buffer.times[-1]
        # self.diff_times_pcd_pose.append(diff)

        # get the closest initial estimation at the pointcloud timestamp
        posei, timestampi, case_type = self.localized_pose_buffer.interpolated_pose_at_time_new(timestamp=timestamp_pcd)

        # posei, timestampi = self.localized_pose_buffer.interpolated_pose_at_time(timestamp=timestamp_pcd,
        #                                                                          delta_threshold_s=DELTA_THRESHOLD_S)
        # posei, timestampi = self.localized_pose_buffer.get_closest_pose_at_time(timestamp=timestamp_pcd,
        #                                                                         delta_threshold_s=DELTA_THRESHOLD_S)
        # this is needed to find localized_pose measurements corresponding to the last received pointcloud
        hz = 4
        # queue_size=3
        time_to_wait = (1/hz)
        if posei is None:
            print('Caution: no initial estimation found por pcd')
            print('Waiting need a delay')
            print(30*'ERROR posei ')
            rospy.wait_for_message(INITIAL_ESTIMATED_POSE, Odometry)
            return
        # we are on the edge! must wait till case_type==2
        if (case_type == 0) or (case_type == 1):
            print("FIRST ELEMENT OR LAST ELEMENT INTERPOLATION DETECTED. SLEEPING")
            # since we are on the rigth of the queue of initial estimated poses,
            # we wait till the next one and wait for more pcds to stay at the queue
            rospy.wait_for_message(INITIAL_ESTIMATED_POSE, Odometry)
            return

        # the initial estimation at that time is:
        T0i = posei.T()
        print('PERFORMING LOCALIZATION REFINEMENT!')
        print('***************')
        # now we have a global transformation and a pointcloud at timestamp_pcd (the closest to timestamp_pcd)
        # perform global localization (given an initial guess, compute the final prior estimation)
        T0i_ = self.global_map.perform_global_localization_of_scan_i(local_scan=pcd,
                                                                     initial_guess=T0i,
                                                                     show=False)
        # publish the estimation
        self.publish_prior_information_pose(T=T0i_, timestamp=timestamp_pcd)
        self.all_initial_estimations.append(T0i)
        self.all_refined_estimations.append(T0i_)
        end = time.time()
        print('TOTAL TIME FOR GLOBAL LOALIZATION: ', end-start)
        return

    def publish_prior_information_pose(self, T, timestamp):
        """
        Publish the estimation found on any pose close to the published time.
        """
        if T is None:
            return
        print('Publishing last pose:')
        position = T.pos()
        orientation = T.Q()
        msg = Odometry()
        msg.header.stamp = rospy.Time.from_sec(timestamp) # rospy.Time.now()
        # caution: this index in graph is directly related to the index
        # in the grapshslam
        msg.header.frame_id = "map" # str(index_in_graph) #"map"
        msg.child_frame_id = "odom"
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
        all_initial_estimations = []
        all_refined_estimations = []
        for T0i in self.all_initial_estimations:
            all_initial_estimations.append(T0i.pos())
        all_initial_estimations = np.array(all_initial_estimations)

        for T0j in self.all_refined_estimations:
            all_refined_estimations.append(T0j.pos())
        all_refined_estimations = np.array(all_refined_estimations)

        ax1.clear()
        if len(all_initial_estimations) > 0:
            ax1.scatter(all_initial_estimations[:, 0], all_initial_estimations[:, 1], marker='.', color='blue')

        if len(all_refined_estimations) > 0:
            ax1.scatter(all_refined_estimations[:, 0],
                        all_refined_estimations[:, 1], marker='.', color='red')
        ax1.legend()
        canvas1.print_figure('plots/run_scanmatcher_to_map_plot.png', bbox_inches='tight', dpi=300)


        ax2.clear()
        diff_times_pcd_pose = np.array(self.diff_times_pcd_pose)
        if len(diff_times_pcd_pose) > 0:
            ax2.plot(diff_times_pcd_pose, marker='.', color='blue', label='Initial pose estimation idff times')

        # pcd_times = np.array(self.pcdbuffer.times)
        # if len(pcd_times) > 0:
        #     ax2.plot(pcd_times, marker='.', color='red', label='Pointcloud times')
        ax2.legend()
        canvas2.print_figure('plots/run_scanmatcher_to_map_plot2_times.png', bbox_inches='tight', dpi=300)

        # print('Plotting info on times')
        # ax2.clear()
        # pcd_times = np.array(self.pcdbuffer.times) - self.start_time
        # if len(pcd_times) > 0:
        #     ax2.plot(pcd_times, marker='.', color='blue', label='Pointcloud times')
        # initial_estimation_poses_buffer_times = np.array(self.localized_pose_buffer.times) - self.start_time
        # if len(initial_estimation_poses_buffer_times) > 0:
        #     ax2.plot(initial_estimation_poses_buffer_times, marker='.', color='black', label='Localized pose times')
        # ax2.legend()
        # ax2.grid()
        # canvas2.print_figure('plots/run_scanmatcher_to_map_plot2_times.png', bbox_inches='tight', dpi=300)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = GlobalScanMatchingROSNode()
    node.run()


