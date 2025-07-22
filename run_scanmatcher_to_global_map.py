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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from observations.posesbuffer import PosesBuffer
import time
import open3d as o3d
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt
from collections import deque
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from config import PARAMETERS

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_title('SCANMATCHING path positions')
canvas1 = FigureCanvas(fig1)

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.set_title('Computation time scanmatching')
canvas2 = FigureCanvas(fig2)

# MAP_DIRECTORY = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO5-2025-06-16-17-53-54'
# MAP_DIRECTORY = 'map_data'
# MAP_FILENAME = 'map_data/global_map.pcd'
MAP_FILENAME = PARAMETERS.config.get('scanmatcher_to_map').get('map_filename')
DELTA_THRESHOLD_S = 1.0 # for interpolation on localized_pose
# CAUTION: this topic must be subscribed to the /ouster/points (high rate) topic
# POINTCLOUD_TOPIC = '/ouster/points_low_rate'
POINTCLOUD_TOPIC = PARAMETERS.config.get('scanmatcher_to_map').get('pointcloud_input_topic')

# INITIAL ESTIMATION POSE, this is the output of the run_graph_localizer algorithm
# INITIAL_ESTIMATED_POSE = '/localized_pose'
INITIAL_ESTIMATED_POSE = PARAMETERS.config.get('scanmatcher_to_map').get('localized_pose_input_topic')

# WE ARE PUBLISHING THE PRIOR ESTIMATION HERE, as a result of the localization in the map
# MAP_SM_GLOBAL_POSE_TOPIC = '/map_sm_global_pose'
MAP_SM_GLOBAL_POSE_TOPIC = PARAMETERS.config.get('scanmatcher_to_map').get('map_sm_prior_output_topic') #'/map_sm_global_pose'

# PUBLISH THE 3D pointcloud global map
# GLOBAL_MAP_TOPIC = '/global_map'
GLOBAL_MAP_TOPIC = PARAMETERS.config.get('scanmatcher_to_map').get('global_map')


class GlobalMap():
    # def __init__(self, map_directory, map_filename):
    def __init__(self, map_filename):
        # 1. Load the global map (PCD)
        # map_filename = map_directory + '/' + map_filename
        print('LOADING MAP: ', map_filename)
        self.global_map = o3d.io.read_point_cloud(map_filename)
        print("[INFO] Global map loaded with", len(self.global_map.points), "points.")
        # 3. Downsample for faster processing the global map
        # voxel_size_global_map = 0.2  # adjust as needed
        # self.global_map.voxel_down_sample(voxel_size_global_map)
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
        # reduce to a local map, reduce the number of points
        # filter a submap of 20x20 centered on the current xy position
        global_map_temp = self.global_map
        x = initial_guess.array[0][3]
        y = initial_guess.array[1][3]

        # Filter a submap of the  global map, must be saved as parameter
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
        # 6. ICP registration (use as NDT substitute)
        reg_result = o3d.pipelines.registration.registration_icp(
            local_scan.pointcloud, global_map_temp, max_correspondence_distance=8.0,
            init=initial_guess.array,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        print("[RESULT] Transformation matrix:\n", reg_result.transformation)
        print("[INFO] Fitness:", reg_result.fitness, "Inlier RMSE:", reg_result.inlier_rmse)
        print("[RESULT]", reg_result)

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
        # store the localized pose in deque fashion
        self.localized_pose_buffer = PosesBuffer(maxlen=5000)
        # store the received pointclouds
        self.lidar_queue = deque(maxlen=100)
        # LOAD THE MAP
        print('Loading MAP: ', MAP_FILENAME)
        self.global_map = GlobalMap(map_filename=MAP_FILENAME)
        print('Map loaded!')

        print("Voxelizando la nube for publishing...")
        pointcloud_publish = self.global_map.global_map.voxel_down_sample(voxel_size=0.5)
        print(f"Tamaño nube voxelizada: {np.asarray(pointcloud_publish.points).shape}")

        # ROS STUFFF
        print('Initializing global scanmatching node!')
        rospy.init_node('scanmatcher_to_map_node')
        print('Subscribing to PCD, GNSS')
        print('WAITING FOR MESSAGES!')
        print('NODE PARAMETERS: ')
        print(PARAMETERS.config.get('scanmatcher_to_map'))
        print('TOPICS: ')
        print('INPUT POINTCLOUD TOPIC: ', POINTCLOUD_TOPIC)
        print('INPUT INITIAL ESTIMATED LOCALIZED POSE TOPIC: ', INITIAL_ESTIMATED_POSE)
        print('OUTPUT SCANMATCHED POSE, LOCALIZED TO MAP POSE TOPIC: ', MAP_SM_GLOBAL_POSE_TOPIC)
        print('OUTPUT GLOBAL PCD TOPIC: ', GLOBAL_MAP_TOPIC)

        # Subscriptions to the pointcloud topic and to the
        # CAution!!! There must be a delay between the received pointcloud and the /localized_pose trajectory.
        # In particular, the pointcloud times must be retarded with respect to the /localized_pose
        # Caution: queue_size must be 1 or 2 approxx
        rospy.Subscriber(POINTCLOUD_TOPIC, PointCloud2, self.pc_callback, queue_size=10)
        # subscription to the current localized pose
        rospy.Subscriber(INITIAL_ESTIMATED_POSE, Odometry, self.localized_pose_callback)

        # Set up a timer to periodically update the png plots
        rospy.Timer(rospy.Duration(5), self.plot_timer_callback)

        # Publisher
        self.pub = rospy.Publisher(MAP_SM_GLOBAL_POSE_TOPIC, Odometry, queue_size=10)
        self.map_publisher = rospy.Publisher(GLOBAL_MAP_TOPIC, PointCloud2, queue_size=1, latch=True)
        print('Publishing global pointcloud PCD map in ROS to: ', GLOBAL_MAP_TOPIC)
        self.publish_map(pointcloud_publish)

        # print info.
        self.all_initial_estimations = [] # initial without map
        self.all_refined_estimations = [] # compared to map
        # caution: skipping some of the pointclouds
        self.times_pcd = []
        self.times_pose_buffer = []
        self.diff_times_pcd_pose = []

    def pc_callback(self, msg):
        """
        Get last pcd reading and add to buffer
        """
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        print('Received pointcloud')
        pcd = LidarScan(time=timestamp, pose=None)
        pcd.load_pointcloud_from_msg(msg=msg)
        self.lidar_queue.append(pcd)
        return

    def localized_pose_callback(self, msg):
        """
            Obtain the last estimations on the robot path
            This /localized_pose is used as an initial estimation and usually obtained from the localization node itself.
            This should be the /localized_pose topic, which maintains
            the last localization with all the information, excluding
            the localization with respec to the map.
        """
        print('Received localized pose!!!')
        timestamp = msg.header.stamp.to_sec()
        if self.start_time is None:
            self.start_time = timestamp
        pose = Pose()
        pose.from_message(msg.pose.pose)
        # append data to buffer
        self.localized_pose_buffer.append(pose, timestamp)
        # try to locate a pcd within the last two received poses
        self.find_and_localize()

    def find_and_localize(self):
        """
            Obtain the last estimations on the robot path
            This /localized_pose is used as an initial estimation and usually obtained from the localization node itself.
            This should be the /localized_pose topic, which maintains
            the last localization with all the information, excluding
            the localization with respect to the map.

            To do this we have:
            - a buffer of loacized_poses (the last estimation). This function is called whenever a new localized posed
            ois received. Only the last two values are used.
            - a buffer of pcds. We try to find a pcd between the times of the last localized poses (startgin from the last received pcd)
        """
        print(50 * '_')
        print('Calling global localization method!!!')
        print('localized_pose_buffer size: ', len(self.localized_pose_buffer))
        print('lidar_queue size: ', len(self.lidar_queue))
        print(50 * '_')
        if len(self.localized_pose_buffer) < 2:
            print('Did not receive enough localized initial poses')
            return
        # The following are the initial and final times
        ta = self.localized_pose_buffer.times[-2] # penultim temps
        tb = self.localized_pose_buffer.times[-1] # ultim temps
        pcdini = self.lidar_queue[0] # temps de la primera en buffer
        pcdend = self.lidar_queue[-1] # ultima en buffer
        print('Last two times localized pose: ', ta-self.start_time, tb-self.start_time)
        print('Times buffer lidar buffer: ', pcdini.time - self.start_time, pcdend.time - self.start_time)
        # print('Diff time localized buffer - pcd buffer end: ', tend-pcdend.time)
        print(50 * '_')

        N = len(self.lidar_queue)
        # try to localize on a buffer of saved pcds
        # start with the last received pcd. Always try to keep up with the last received PCD
        # starting with the last received pcd, we try to localize the pcd if an initial estimation /localized_pose
        # has been received
        for i in range(N-1, 0, -1):
            # print(20 * '+')
            # print('In queue, processing in temporal buffer for pcds: ', i)
            # caution, looking on the right, the last
            pcd = self.lidar_queue[i]
            timestamp_pcd = pcd.time
            # This filters rapidly if the pointcloud hascannot localize in this case
            if ta < timestamp_pcd < tb:
                # print('continue timestamp_pcd < tini')
                # try to localize the last received pose
                self.compute_global_scanmatching(pcd, timestamp_pcd=timestamp_pcd)
                break

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
            return False

        # get the closest initial estimation at the pointcloud timestamp
        posei, timestampi, case_type = self.localized_pose_buffer.interpolated_pose_at_time_new(timestamp=timestamp_pcd)

        if posei is None:
            print('Caution: no initial estimation found por pcd')
            print('Waiting need a delay')
            print(30*'ERROR posei ')
            return False
        # we are on the edge! must wait till case_type==2
        if (case_type == 0) or (case_type == 1):
            print("FIRST ELEMENT OR LAST ELEMENT INTERPOLATION DETECTED. SLEEPING")
            return False

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
        #success
        return True

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

    def publish_map(self, pointcloud):
        """
        Publish the global pointcloud map
        """
        print("##### Calling publish_map.")
        points = np.asarray(pointcloud.points)

        # Crea el header
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"

        # Crea la estructura de los campos (x, y, z)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convierte los puntos a una lista de tuplas
        point_list = [tuple(p) for p in points]

        # Crea el mensaje PointCloud2
        pc2_msg = pc2.create_cloud(header, fields, point_list)

        self.map_publisher.publish(pc2_msg)
        print("##### Mapa publicado!!!!!!.")

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

        ax2.legend()
        canvas2.print_figure('plots/run_scanmatcher_to_map_plot2_times.png', bbox_inches='tight', dpi=300)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = GlobalScanMatchingROSNode()
    node.run()


