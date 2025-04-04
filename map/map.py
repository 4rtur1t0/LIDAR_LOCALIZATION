import numpy as np
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.posesarray import PosesArray, ArucoPosesArray, ArucoLandmarksPosesArray


class Map():
    """
    A Map!
    The map is composed by:
    - A list of estimated poses X as the robot followed a path.
    - A LiDAR scan associated to each pose.
    - A number of ARUCO landmarks. This part is optional, however



    A number of methods are included in this class:
    - Plotting the global pointcloud, including the path and landmarks.
    - Plotting the pointcloud

    - Includes a method to localize on the map


    """
    def __init__(self):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.robotpath = None
        self.lidarscanarray = None
        self.landmarks_aruco = None

    def __len__(self):
        return len(self.robotpath)

    def read_data(self, directory):
        """
        Read the estimated path of the robot, landmarks and LiDAR scans.
        the map is formed by a set of poses, each pose associated to a pointcloud.
        no global map is built and stored. Though, the view_map method allows
        """
        self.robotpath = PosesArray()
        # caution: reading the solution referred to the GPS reference system
        self.robotpath.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_gps.csv')
        # Load the LiDAR scan array. Each pointcloud with its associated time.
        # Each lidar scan is associated to a given pose in the robotpath
        self.lidarscanarray = LiDARScanArray(directory=directory)
        self.lidarscanarray.read_parameters()
        self.lidarscanarray.read_data()
        # remove scans without corresponding known poses
        self.lidarscanarray.remove_orphan_lidars(pose_array=self.robotpath)
        # load the scans according to the times, do not load the corresponding pointclouds
        self.lidarscanarray.add_lidar_scans()
        # also load the ARUCO landmarks
        self.landmarks_aruco = ArucoLandmarksPosesArray()
        self.landmarks_aruco.read_data(directory=directory, filename='/robot0/SLAM/solution_graphslam_aruco_landmarks.csv')

    def draw_all_clouds(self):
        self.lidarscanarray.draw_all_clouds()

    def draw_map(self, voxel_size, keyframe_sampling=20, terraplanist=False):
        """
        Possibilities:
        - view path
        - view pointclouds
        - view ARUCO landmarks
        """
        global_transforms = self.robotpath.get_transforms()
        self.lidarscanarray.draw_map(global_transforms=global_transforms,
                                     voxel_size=voxel_size,
                                     radii=[1, 10],
                                     heights=[-2, 1.2],
                                     keyframe_sampling=keyframe_sampling,
                                     terraplanist=terraplanist)

    def localize_with_aruco(self, Tca, aruco_id, **kwargs):
        """
        Performs an initial localization step
        """
        print('INITIAL LOCALIZATION!')
        print('FOUND ARUCO')
        Tlidar_gps = kwargs.get('Tlidar_gps')
        Tlidar_cam = kwargs.get('Tlidar_cam')
        Tgps_lidar = Tlidar_gps.inv()
        # observation of the ARUCO from the gps reference system
        Tgps_aruco = Tgps_lidar * Tlidar_cam * Tca
        # aruco_id = arucoobsarray.get_aruco_id(j)
        # aruco_ids_in_map = self.landmarks_aruco.aruco_ids
        index = np.where(self.landmarks_aruco.aruco_ids == aruco_id)[0]
        print('Found aruco_id at position: ', index)
        Tglobal_aruco = self.landmarks_aruco.values[index[0]]
        Tglobal_aruco = Tglobal_aruco.T()
        print('Found ARUCO, aruco_id, ', aruco_id, 'at global pose: ')
        Tglobal_aruco.print()
        Tgps_robot = Tglobal_aruco*Tgps_aruco.inv()
        print('Robot is localized at: ')
        Tgps_robot.print()
        return Tgps_robot