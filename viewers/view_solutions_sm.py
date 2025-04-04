"""
Visualize the all data easily
"""
from eurocreader.eurocreader import EurocReader
from tools.gpsconversions import gps2utm
from tools.plottools import plot_gps_OSM, plot_gps_points, plot_3D_data
from tools.plottools import plot_xy_data, plot_xyz_data, plot_quaternion_data
import numpy as np
from artelib.vector import Vector
from artelib.euler import Euler
from artelib.homogeneousmatrix import HomogeneousMatrix
import matplotlib.pyplot as plt

def plot_result(df_data, marker):
    plt.plot(df_data['x'], df_data['y'], marker=marker)
    # plt.legend()

def plot_deltas(deltas):
    plt.plot(deltas)


def compute_basic_end_error(df_sm):
    x = df_sm['x'].tolist()
    y = df_sm['y'].tolist()
    z = df_sm['z'].tolist()
    e = np.array([x[-1], y[-1], z[-1]])
    return np.linalg.norm(e)


def compute_deltas_xy(df_data):
    deltas = []
    for i in range(len(df_data)-1):
        dx = df_data['x'][i+1]-df_data['x'][i]
        dy = df_data['y'][i + 1] - df_data['y'][i]
        d = np.linalg.norm(np.array([dx, dy]))
        deltas.append(d)
    deltas = np.array(deltas)
    return deltas



# def transform_gps_data(df_gps):
#     df_gps = gps2utm(df_gps)
#     x = df_gps['x'].tolist()
#     y = df_gps['y'].tolist()
#     angle = np.arctan2(y[1]-y[0], x[1]-x[0])
#     T = HomogeneousMatrix(Vector([0, 0, 0]), Euler([0, 0, angle]))
#     t = []
#     for i in range(len(x)):
#         v = Vector([x[i], y[i], [0], 1.0])
#         vp = T*v
#         t.append(vp)
#     return np.array(t)

def view_gps_data(directory):
    """
    View lat/lng data on 2D. Also, plot on OSM
    """
    try:
        euroc_read = EurocReader(directory=directory)
        df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
        plot_gps_points(df_gps=df_gps, annotate_index=True)
        plot_gps_points(df_gps=df_gps, annotate_error=True)
        # plot_gps_OSM(df_gps=df_gps, save_fig=True, expand=0.001)
    except FileNotFoundError:
        print('NO GPS DATA FOUND')


def view_gps_utm_data(directory):
    """
    View lat/lng data on 2D. Also, plot on OSM
    """
    try:
        euroc_read = EurocReader(directory=directory)
        df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
        df_gps = gps2utm(df_gps)
        plot_xy_data(df_data=df_gps)
        # plot_gps_OSM(df_gps=df_gps, save_fig=True, expand=0.001)
    except FileNotFoundError:
        print('NO GPS DATA FOUND')


# def prepare_experiment_data(directory, scanmatcher_file):
#     """
#     Use, gps timestamps as base
#     """
#     # Read LiDAR times
#     euroc_read = EurocReader(directory=directory)
#     df_gps_times = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
#     gps_times = df_gps_times['#timestamp [ns]'].to_numpy()
#     # Read LiDAR transformation (between each of the previous times)
#     df_lidar_global = euroc_read.read_csv(filename='/robot0/scanmatcher/' + scanmatcher_file)
#     lidar_times = df_lidar_global['#timestamp [ns]'].to_numpy()
#     lidar_times = euroc_read.get_closest_times(master_sensor_times=gps_times, sensor_times=lidar_times)
#     df_lidar_global = euroc_read.get_df_at_times(df_data=df_lidar_global, time_list=lidar_times)
#
#     return df_lidar_global

# def prepare_experiment_data(directory, scanmatcher_file):
#     """
#     Use, lidar timestamps as base, then:
#         - for each lidar, find the closest GPS.
#         - or for each lidar, interpolate a GPS value.
#     """
#     # Read LiDAR times
#     euroc_read = EurocReader(directory=directory)
#     # Read LiDAR transformation (between each of the previous times)
#     df_lidar_global = euroc_read.read_csv(filename='/robot0/scanmatcher/' + scanmatcher_file)
#     lidar_times = df_lidar_global['#timestamp [ns]'].to_numpy()
#
#     df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
#     gps_times = df_gps['#timestamp [ns]'].to_numpy()
#     # Read LiDAR transformation (between each of the previous times)
#     # df_lidar_global = euroc_read.read_csv(filename='/robot0/scanmatcher/' + scanmatcher_file)
#     # lidar_times = df_lidar_global['#timestamp [ns]'].to_numpy()
#     gps_times = euroc_read.get_closest_times(master_sensor_times=lidar_times, sensor_times=gps_times)
#     df_gps = euroc_read.get_df_at_times(df_data=df_gps, time_list=gps_times)
#
#     return df_lidar_global



if __name__ == "__main__":
    directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
    filename = '/robot0/scanmatcher/scanmatcher_gps_global.csv'
    # ground truth
    euroc_read = EurocReader(directory=directory)
    df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
    df_gps = gps2utm(df_gps=df_gps)
    # df_sm = prepare_experiment_data(directory, filename)
    df_sm = euroc_read.read_csv(filename=filename)

    # CAUTION: if transform GPS coordinates so that they have the same origin

    df_sm = df_sm[0:50]
    df_gps = df_gps[0:25]

    # handle times
    gps_times = df_gps['#timestamp [ns]'].to_numpy()
    lidar_times = df_sm['#timestamp [ns]'].to_numpy()
    experiment_base_reference_time = lidar_times[0]
    # convert experiment times to seconds
    gps_times = (gps_times-experiment_base_reference_time)/1e9
    lidar_times = (lidar_times - experiment_base_reference_time)/ 1e9


    plt.figure(0)
    plt.plot(gps_times)
    plt.plot(lidar_times)
    plt.legend(['GPS TIMES', 'LIDAR TIMES'])
    plt.show()

    # plot XY global trajectory
    plt.figure(1)
    plot_result(df_gps, marker='o')
    plot_result(df_sm, marker='.')
    plt.legend(['GPS', 'ICP_pointpoint'])
    plt.show()

    # Now compute deltas
    deltas_gps = compute_deltas_xy(df_data=df_gps)
    deltas_sm = compute_deltas_xy(df_data=df_sm)
    # error
    e = deltas_gps - deltas_sm

    plt.figure(1)
    plot_deltas(deltas_gps)
    plot_deltas(deltas_sm)
    plot_deltas(e)
    plt.legend(['DELTAS GPS', 'DELTAS SM', 'ERROR'])
    plt.show()

    print('ERRORS: ')
    print('MEAN ERROR: (m)', np.mean(e))
    print('STD ERROR: (m)', np.std(e))



    # plot_xyz_data(df_data=df_sm, title='SM ICP POINT POINT')
    # plot_3D_data(df_data=df_sm)
    # plot_xy_data(df_data=df_sm, title='SM ICP POINT POINT')
    # error = compute_basic_end_error(df_sm)
    # print('SM ICP point point error: ', error)

    # df_sm = euroc_read.read_csv(filename='/robot0/scanmatcher/O2/scanmatcher_global_0.3_icppointplane.csv')
    # plot_xyz_data(df_data=df_sm, title='SM ICP POINT POINT')
    # plot_3D_data(df_data=df_sm)
    # plot_xy_data(df_data=df_sm, title='SM ICP POINT PLANE')
    # error = compute_basic_end_error(df_sm)
    # print('SM ICP point plane error: ', error)
    #
    # df_sm = euroc_read.read_csv(filename='/robot0/scanmatcher/scanmatcher_global_icp2planes.csv')
    # plot_xyz_data(df_data=df_sm, title='SM ICP POINT POINT')
    # plot_3D_data(df_data=df_sm)
    # plot_xy_data(df_data=df_sm, title='SM ICP 2PLANES')
    # error = compute_basic_end_error(df_sm)
    # print('SM ICP 2 planeS error: ', error)


