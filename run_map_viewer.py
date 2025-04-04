"""
    Visualize map from known/ground truth trajectory and LIDAR.

    Author: Arturo Gil.
    Date: 03/2024

    TTD: Save map for future use in MCL localization.
         The map can be now saved in PCL format in order to use o3D directly.
         Also, the map can be saved in a series of pointclouds along with their positions, however path planning using,
         for example, PRM, may not be direct
"""
from map.map import Map


def map_viewer():
    # Read the final transform (i.e. via GraphSLAM)
    # You may be using different estimations to build the map: i.e. scanmatching or the results from graphSLAM
    # select as desired
    directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'

    # use, for example, 1 out of 5 LiDARS to build the map
    keyframe_sampling = 5
    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    voxel_size = 0.3
    maplidar = Map()
    maplidar.read_data(directory=directory)
    # visualize the clouds relative to the LiDAR reference frame
    # maplidar.draw_all_clouds()
    # visualize the map on the UTM reference frame
    maplidar.draw_map(terraplanist=False, keyframe_sampling=keyframe_sampling, voxel_size=voxel_size)
    # maplidar.build_map()


if __name__ == '__main__':
    map_viewer()

