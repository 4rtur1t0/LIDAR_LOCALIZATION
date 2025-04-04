import time
import numpy as np
# from artelib.euler import Euler
# from artelib.homogeneousmatrix import HomogeneousMatrix
import open3d as o3d
import copy
# from config import ICP_PARAMETERS


class LiDARScan():
    def __init__(self, directory, scan_time, parameters):
        # directory
        self.directory = directory
        self.scan_time = scan_time
        # the pointcloud
        self.pointcloud = None  # o3d.io.read_point_cloud(filename)
        # voxel sizes
        # self.voxel_size = parameters.get('voxel_size', None)
        self.voxel_size_normals = parameters.get('voxel_size_normals', None)
        self.max_nn_estimate_normals = parameters.get('max_nn_estimate_normals', None)
        # filter
        self.min_reflectivity = parameters.get('min_reflectivity', None)
        self.min_radius = parameters.get('min_radius', None)
        self.max_radius = parameters.get('max_radius', None)
        self.min_height = parameters.get('min_height', None)
        self.max_height = parameters.get('max_height', None)

    def load_pointcloud(self):
        filename = self.directory + '/robot0/lidar/data/' + str(self.scan_time) + '.pcd'
        print('Reading pointcloud: ', filename)
        # Load the original complete pointcloud
        self.pointcloud = o3d.io.read_point_cloud(filename)

    def save_pointcloud(self):
        filename = self.directory + '/robot0/lidar/dataply/' + str(self.scan_time) + '.ply'
        print('Saving pointcloud: ', filename)
        # Load the original complete pointcloud
        o3d.io.write_point_cloud(filename, self.pointcloud)

    def filter_points(self):
        self.down_sample()
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
        target_temp = copy.deepcopy(other.pointcloud)
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
        print('Removing pointclouds from memory (filtered, planes, fpfh): ')
        del self.pointcloud
        # del self.pointcloud_filtered
        # del self.pointcloud_fpfh
        # del self.pointcloud_ground_plane
        # del self.pointcloud_non_ground_plane
        self.pointcloud = None
        # self.pointcloud_filtered = None
        # self.pointcloud_ground_plane = None
        # self.pointcloud_non_ground_plane = None
        # self.pointcloud_fpfh = None







