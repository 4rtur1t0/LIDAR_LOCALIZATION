"""
Visualize the LIDAR point-clouds in the EUROC directory.
"""
from artelib.homogeneousmatrix import HomogeneousMatrix
from eurocreader.eurocreader import EurocReader
from keyframemanager.keyframemanager import KeyFrameManager

#
# def main():
#     directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
#     # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
#     # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
#     # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-03-20-13-14-41'
#     # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
#     euroc_read = EurocReader(directory=directory)
#     scan_times = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
#     scan_times = scan_times['#timestamp [ns]'].to_numpy()
#     # create KeyFrameManager
#     keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times,
#                                        voxel_size=None)
#     keyframe_manager.draw_all_clouds(sample=5, max_dist=25, max_height=1.5)


def view_point_clouds(directory, keyframe_sampling):
    """
    View the map (visualize_map_online) or build it.
    When building it, an open3D kd-tree is obtained, which can be saved to a file (i.e.) a csv file.
    Also, the map can be viewed as a set of poses (i.e. x,y,z, alpha, beta, gamma) at certain timestamps associated to
    a scan reading at that time.
    """
    euroc_read = EurocReader(directory=directory)
    scan_times = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
    scan_times = scan_times['#timestamp [ns]'].to_numpy()

    # build identity transformations
    identity_transforms = []
    for i in range(0, len(scan_times), keyframe_sampling):
        identity_transforms.append(HomogeneousMatrix())

    # use, for example, voxel_size=0.2. Use voxel_size=None to use full resolution
    voxel_size = None
    keyframe_manager = KeyFrameManager(directory=directory, scan_times=scan_times, voxel_size=voxel_size)
    # OPTIONAL: visualize resulting map
    keyframe_manager.add_keyframes(keyframe_sampling=keyframe_sampling)
    # keyframe_manager.load_pointclouds()
    # caution: only visualization. All points are kept by the visualization window
    # caution: the global transforms correspond to the scan_times
    keyframe_manager.visualize_map_online(global_transforms=identity_transforms, radii=[0.5, 30.0], clear=True)
    # the build map method actually returns a global O3D pointcloud
    # pointcloud_global = keyframe_manager.build_map(global_transforms=identity_transforms,
    #                                                keyframe_sampling=keyframe_sampling, radii=[0.5, 10.0])
#

if __name__ == "__main__":
    # OUTDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
    directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-03-20-13-14-41'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
    # INDOOR
    directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
    # mixed INDOOR/OUTDOOR
    #directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
    view_point_clouds(directory=directory, keyframe_sampling=5)
