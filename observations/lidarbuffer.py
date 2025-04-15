import time
import bisect
from collections import deque
import open3d as o3d
import copy

import gc

from config import PARAMETERS
import sensor_msgs.point_cloud2 as pc2
import numpy as np


class LidarBuffer:
    def __init__(self, maxlen=20):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.times = deque(maxlen=maxlen)
        self.pointclouds = deque(maxlen=maxlen)
        # stores the pcds that have been processed
        # self.processed = deque(maxlen=maxlen)
        self.show_registration_result = False

    def from_msg(self):
        return

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        return self.pointclouds[item]

    def get(self, item):
        return self.pointclouds[item]

    def append(self, pcd, time):
        """
        append a pcd and time to arrays
        """
        self.pointclouds.append(pcd)
        self.times.append(time)

    def popleft(self):
        self.pointclouds[0].unload_pointcloud()
        self.pointclouds.popleft()
        self.times.popleft()

    def get_time(self, index):
        """
        Get the time for a corresponding index
        """
        return self.times[index]

    def get_times(self):
        """
        Get all the scan times
        """
        return self.times

    def get_pcds(self):
        return self.pointclouds

    def get_at_exact_time(self, timestamp):
        """
        Get the pointcloud found at a exact, particular, timestamp
        """
        index = bisect.bisect_left(self.pointclouds, timestamp)
        if index < len(self.times) and self.times[index] == timestamp:
            print(f"Element {timestamp} found at index {index}")
            return self.pointclouds[index]
        else:
            return None

    # def unload_pointcloud(self, i):
    #     self.pointclouds[i].unload_pointcloud()
    #
    # def pre_process(self, index):
    #     self.pointclouds[index].pre_process(method=self.method)
    #
    # def filter_points(self, index):
    #     self.pointclouds[index].filter_points()
    #
    # def estimate_normals(self, index):
    #     self.pointclouds[index].estimate_normals()
    #
    # def draw_cloud(self, index):
    #     self.pointclouds[index].draw_cloud()


class LidarScan():
    def __init__(self, time, pose, directory=None):
        # directory
        self.directory = directory
        # time of the scan
        self.time = time
        # pose at which the scan was captured (maybe odometry)
        self.pose = pose
        # the pointcloud
        self.pointcloud = None  # o3d.io.read_point_cloud(filename)
        # voxel sizes
        self.voxel_size = PARAMETERS.config.get('scanmatcher').get('voxel_size', None)
        self.voxel_size_normals = PARAMETERS.config.get('scanmatcher').get('normals').get('voxel_size_normals', None)
        self.max_nn_estimate_normals = PARAMETERS.config.get('scanmatcher').get('normals').get('max_nn_normals', None)
        # filter
        self.min_reflectivity = PARAMETERS.config.get('scanmatcher').get('min_reflectivity', None)
        self.min_radius = PARAMETERS.config.get('scanmatcher').get('min_radius', None)
        self.max_radius = PARAMETERS.config.get('scanmatcher').get('max_radius', None)
        self.min_height = PARAMETERS.config.get('scanmatcher').get('min_height', None)
        self.max_height = PARAMETERS.config.get('scanmatcher').get('max_height', None)

    def load_pointcloud_from_msg(self, msg):
        self.time = msg.header.stamp.to_sec()
        cloud = np.array([p[:3] for p in pc2.read_points(msg, skip_nans=True)])
        self.pointcloud = o3d.geometry.PointCloud()
        self.pointcloud.points = o3d.utility.Vector3dVector(cloud)

    def load_pointcloud(self):
        filename = self.directory + '/robot0/lidar/data/' + str(self.time) + '.pcd'
        print('Reading pointcloud: ', filename)
        # Load the original complete pointcloud
        self.pointcloud = o3d.io.read_point_cloud(filename)

    def save_pointcloud(self):
        filename = self.directory + '/robot0/lidar/dataply/' + str(self.time) + '.ply'
        print('Saving pointcloud: ', filename)
        # Load the original complete pointcloud
        o3d.io.write_point_cloud(filename, self.pointcloud)

    def filter_points(self):
        # self.down_sample()
        self.filter_radius()
        self.filter_height()

    def down_sample(self, voxel_size=None):
        if voxel_size is None:
            return
        self.pointcloud = self.pointcloud.voxel_down_sample(voxel_size=voxel_size)

    def filter_radius(self, radii=None):
        if radii is None:
            self.pointcloud = self.filter_by_radius(self.min_radius, self.max_radius)
        else:
            self.pointcloud = self.filter_by_radius(radii[0], radii[1])

    def filter_height(self, heights=None):
        if heights is None:
            self.pointcloud = self.filter_by_height(-120.0, 120.0)
        else:
            self.pointcloud = self.filter_by_height(heights[0], heights[1])

    def filter_by_radius(self, min_radius, max_radius):
        points = np.asarray(self.pointcloud.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        r2 = x ** 2 + y ** 2
        # idx = np.where(r2 < max_radius ** 2) and np.where(r2 > min_radius ** 2)
        idx2 = np.where((r2 < max_radius ** 2) & (r2 > min_radius ** 2))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))

    def filter_by_height(self, min_height, max_height):
        points = np.asarray(self.pointcloud.points)
        [x, y, z] = points[:, 0], points[:, 1], points[:, 2]
        idx2 = np.where((z > min_height) & (z < max_height))
        return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx2]))

    def estimate_normals(self):
        self.pointcloud.estimate_normals(
             o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size_normals,
                                                  max_nn=self.max_nn_estimate_normals))

    def transform(self, T):
        return self.pointcloud.transform(T)

    def draw_cloud(self):
        # o3d.visualization.draw_geometries([self.pointcloud],
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
        #                                   up=[-0.0694, -0.9768, 0.2024])
        o3d.visualization.draw_geometries([self.pointcloud])

    def draw_cloud_visualizer(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(self.pointcloud)
        try:
            while True:
                if not vis.poll_events():
                    print("Window closed by user")
                    break
                vis.update_renderer()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Interrupted by user")
        vis.destroy_window()

    def draw_registration_result(self, other, transformation):
        source_temp = copy.deepcopy(self.pointcloud)
        target_temp = copy.deepcopy(other.pcd)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 0, 1])
        source_temp.transform(transformation)
        # o3d.visualization.draw_geometries([source_temp, target_temp],
        #                                   zoom=1.0,
        #                                   front=[0, 0, 10],
        #                                   lookat=[0, 0, 0],
        #                                   up=[0, 0, 1])
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def unload_pointcloud(self):
        """
        Remove and collect memory garbage.
        Delete the poincloud actively.
        """
        print('Removing pointclouds from memory (filtered, planes, fpfh): ')
        del self.pointcloud
        self.pointcloud = None
        gc.collect()













