"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.


    This node subscribes to the localized_pose topic.
    The localized_pose topic is initially published by the localization node.
    The initial pose is used to find a number of close pointclouds in the map. A registration is then performed
    As a result, we end up having another prior3Dfactor observation on the state X(i)

"""
from collections import deque
import rospy
# import numpy as np
# from graphSLAM.helper_functions import update_sm_observations, update_odo_observations, \
#     filter_and_convert_gps_observations, update_gps_observations, update_aruco_observations
# from map.map import Map
from nav_msgs.msg import Odometry
# from observations.gpsbuffer import GPSBuffer, GPSPosition
from observations.lidarbuffer import LidarBuffer, LidarScan
from observations.posesbuffer import PosesBuffer, Pose
# from scanmatcher.scanmatcher import ScanMatcher
from sensor_msgs.msg import PointCloud2
# from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from graphSLAM.graphSLAM import GraphSLAM
# from artelib.homogeneousmatrix import HomogeneousMatrix
# from artelib.vector import Vector
# from artelib.euler import Euler
# from config import PARAMETERS
# from tools.gpsconversions import gps2utm
from observations.posesbuffer import PosesBuffer
import time
import open3d as o3d
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
# from lidarscanarray.lidarscanarray import LiDARScanArray
import matplotlib.pyplot as plt
# from observations.posesarray import PosesArray

# from config import PARAMETERS


fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)


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

    # def read_data_tum(self, directory, filename='/robot0/SLAM/data_poses_tum.txt'):
    #     """
    #     Read the estimated path of the robot in tum format
    #     """
    #     robotpath = PosesBuffer()
    #     robotpath.read_data_tum(directory=directory, filename=filename)
    #     positions = robotpath.get_positions()
    #     return np.array(positions)

    def plot_result(self, estimated_positions):
        plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], marker='o', color='blue', label='estimated positions')
        # plt.plot(self.ground_truth_path[:, 0], self.ground_truth_path[:, 1], marker='.', color='black', label='ground_truth')
        # plt.legend()
        plt.show(block=False)
        plt.pause(interval=0.1)

    def perform_global_localization_of_scan_i(self, local_scan, initial_guess, show=False):
        voxel_size_local_scan = 0.2

        print("[INFO] Local scan loaded with", len(local_scan.pointcloud.points), "points.")

        # local_scan.filter_height(heights=heights)
        local_scan.down_sample(voxel_size=voxel_size_local_scan)
        local_scan.filter_radius(radii=[0.5, 12.0])
        local_scan.estimate_normals(voxel_size_normals=voxel_size_local_scan, max_nn_estimate_normals=50)
        print("[INFO] Local scan filtered, normals and down_sampled at", len(local_scan.pointcloud.points), "points.")
        # 2. Load or simulate a new LiDAR scan (local point cloud)
        # local_scan = o3d.io.read_point_cloud("local_scan.pcd")

        # 6. ICP registration (use as NDT substitute)
        reg_result = o3d.pipelines.registration.registration_icp(
            local_scan.pointcloud, self.global_map, max_correspondence_distance=5.0,
            init=initial_guess.array,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        print("[RESULT] Transformation matrix:\n", reg_result.transformation)
        print("[INFO] Fitness:", reg_result.fitness, "Inlier RMSE:", reg_result.inlier_rmse)

        if show:
            # 7. Visualize result
            local_scan.transform(reg_result.transformation)
            # filter height of result
            points = np.asarray(self.global_map.points)
            # [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
            idx2 = np.where((points[:, 2] > -1.0) & (points[:, 2] < 1.5))
            global_map_filtered=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))
            global_map_filtered.paint_uniform_color([1.0, 0, 0])
            local_scan.pointcloud.paint_uniform_color([0, 0, 1.0])
            # o3d.visualization.draw_geometries([local_scan.pointcloud])
            o3d.visualization.draw_geometries([global_map_filtered, local_scan.pointcloud])
        return HomogeneousMatrix(reg_result.transformation)

#
# def map_localization():
#     map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
#     global_map = GlobalMap(map_directory=map_directory, map_filename='global_map.pcd')
#
#     # Load the trajectory!!
#     trajectory_directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
#     # trajectory_directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
#     #trajectory_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17/'
#     starting_index_in_trajectory = 40
#     end_index_in_trajectory = 500
#
#     # read the local observations
#     lidarscanarray = LiDARScanArray(directory=trajectory_directory)
#     lidarscanarray.load_parameters(parameters=PARAMETERS.config.get('scanmatcher'))
#     lidarscanarray.read_data()
#     lidarscanarray.add_lidar_scans()
#
#     # 5. Initial guess (identity or prior pose)
#     # starting from pose zero
#     estimated_positions = []
#     T0 = HomogeneousMatrix()
#     show = False
#     for i in range(starting_index_in_trajectory, end_index_in_trajectory):
#         # if i> 80:
#         #     show = True
#         print('Processing scan', i)
#         print('Localize scan i against the global map')
#         print('A valid initial estimation must be provided')
#         local_scan = lidarscanarray.get(i)
#         local_scan.load_pointcloud()
#         T0 = global_map.perform_global_localization_of_scan_i(local_scan=local_scan,
#                                                               initial_guess=T0,
#                                                               show=show)
#         estimated_positions.append(T0.pos())
#         # print(T0)
#         global_map.plot_result(np.array(estimated_positions))



class GlobalScanMatchingROSNode:
    def __init__(self):
        self.start_time = None
        # the lidar buffer
        self.pcdbuffer = LidarBuffer(maxlen=100)
        self.times_lidar = []
        # store odometry in deque fashion
        self.localized_pose_buffer = PosesBuffer(maxlen=100)
        # this stores the indices in graphslam
        self.localized_pose_buffer_indices = deque(maxlen=100)
        # stores the indexes X(i) of the poses that have been processed
        self.localized_pose_buffer_processed_indices = set()

        # LOAD THE MAP
        print('Loading MAP')
        # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
        # self.map = Map()
        # self.map.read_data(directory=directory)
        #
        # # the scanmatching object
        # self.scanmatcher = ScanMatcher()

        map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
        self.global_map = GlobalMap(map_directory=map_directory, map_filename='global_map.pcd')

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        rospy.init_node('scanmatcher_to_map_node')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')

        # Subscriptions to the pointcloud topic and to the
        # current localized pose
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback)
        rospy.Subscriber(INITIAL_ESTIMATED_POSE, Odometry, self.localized_pose_callback)

        # Set up a timer to periodically update the graph
        t1 = 0.5
        rospy.Timer(rospy.Duration(secs=0, nsecs=int(t1*1e9)), self.compute_global_scanmatching_reversed)
        rospy.Timer(rospy.Duration(1), self.plot_timer_callback)
        # Publisher
        self.pub = rospy.Publisher(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, queue_size=10)

        # print info.
        self.all_initial_estimations = [] # initial without map
        self.all_refined_estimations = [] # compared to map
        # self.prior_estimations = []
        # self.processed_map_poses = []
        # caution: skipping some of the pointclouds
        self.pointcloud_number = 0
        self.pointcloud_skip = 2

    def pc_callback(self, msg):
        """
        Get last pcd reading and append to buffer.
        """
        self.pointcloud_number += 1
        if self.pointcloud_number % self.pointcloud_skip != 0:
            return
        start_time = time.time()
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        self.times_lidar.append(timestamp)
        print('Received pointcloud')
        pcd = LidarScan(time=timestamp, pose=None)
        pcd.load_pointcloud_from_msg(msg=msg)
        self.pcdbuffer.append(pcd=pcd, time=timestamp)

        end_time = time.time()
        print(30 * '+')
        print('Received and loaded pointcloud')
        print(f"pc_callback time:, {end_time - start_time:.4f} seconds")
        print(30 * '+')
        return

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
        pose = Pose()
        pose.from_message(msg.pose.pose)
        self.localized_pose_buffer.append(pose, timestamp)
        self.localized_pose_buffer_indices.append(int(msg.header.frame_id))
        # yes, the current initial estimation has not been processed yet
        # self.localized_pose_buffer_processed.append(False)

    def compute_global_scanmatching_reversed(self, event):
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
        if len(self.pcdbuffer) == 0:
            print("\033[91mCaution!!! No PCD received yet.\033[0m")
            return
        if len(self.localized_pose_buffer) == 0:
            print("\033[91mCaution!!! No initial /localized_pose received yet.\033[0m")
            return
        n = 0
        # publish_prior_estimations = []
        # for each existing state k
        # fix current buffer length
        N = len(self.localized_pose_buffer)
        # caution! reversed
        for k in range(N):
            # caution: this stores the indices received, which correspond to
            # the indices in the graph: important for later publish
            index_i = self.localized_pose_buffer_indices[k]
            # avoid processing the index in graphslam again
            if index_i in self.localized_pose_buffer_processed_indices:
                continue
            #############################################################
            # get the initial prior estate X(i)
            # caution, here, the current index is also stored. The index corresponds to the index in the
            # graph
            ############################################################
            posei = self.localized_pose_buffer[k]
            timestampi = self.localized_pose_buffer.times[k]
            T0i = posei.T()
            # get the closest received timestamp within 1 second.
            pcdi, timestamp_pcd = self.pcdbuffer.get_closest_to_time(timestamp=timestampi, delta_threshold_s=0.2)
            if pcdi is None:
                continue
            print('PERFORMING LOCALIZATION REFINEMENT!')
            print('***************')
            # now we have a global transformation and a pointcloud at timestamp_pcd (the closest to timestamp_pcd)
            # perform global localization (given an initial guess, compute the final prior estimation)
            T0i_ = self.global_map.perform_global_localization_of_scan_i(local_scan=pcdi,
                                                                       initial_guess=T0i,
                                                                       show=False)
            self.publish_prior_information_pose(T=T0i_,
                                                index_in_graph=index_i,
                                                timestamp=timestampi)
            self.localized_pose_buffer_processed_indices.add(index_i)
            # n += 1
            # store for printing
            self.all_initial_estimations.append(T0i)
            self.all_refined_estimations.append(T0i_)
            # yes! a single iteration that is successful
            return


    def publish_prior_information_pose(self, T, index_in_graph, timestamp):
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
        msg.header.frame_id = str(index_in_graph) #"map"
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


        # ax2.clear()
        # initial_poses_times = np.array(self.initial_estimation_poses_buffer.times)/
        # if len(initial_poses_times) > 0:
        #     ax2.plot(initial_poses_times, marker='.', color='blue', label='Initial pose estimation times')
        #
        # pcd_times = np.array(self.pcdbuffer.times)
        # if len(pcd_times) > 0:
        #     ax2.plot(pcd_times, marker='.', color='red', label='Pointcloud times')
        # ax2.legend()
        # canvas2.print_figure('plots/run_scanmatcher_to_map_plot2_times.png', bbox_inches='tight', dpi=300)

        print('Plotting info on times')
        ax2.clear()
        pcd_times = np.array(self.pcdbuffer.times) - self.start_time
        if len(pcd_times) > 0:
            ax2.plot(pcd_times, marker='.', color='blue', label='Pointcloud times')
        initial_estimation_poses_buffer_times = np.array(self.localized_pose_buffer.times) - self.start_time
        if len(initial_estimation_poses_buffer_times) > 0:
            ax2.plot(initial_estimation_poses_buffer_times, marker='.', color='black', label='Localized pose times')
        ax2.legend()
        ax2.grid()
        canvas2.print_figure('plots/run_scanmatcher_to_map_plot2_times.png', bbox_inches='tight', dpi=300)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    filename = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17.bag'
    node = GlobalScanMatchingROSNode()
    node.run()


