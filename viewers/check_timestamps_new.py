"""
Visualize delta_t for each sensor.
"""
from eurocreader.eurocreader import EurocReader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.gpsarray import GPSArray
from observations.posesarray import PosesArray


def computed_distance_travelled(df_sol):
    d = 0
    for i in range(len(df_sol) - 1):
        dx = df_sol['x'].iloc[i + 1] - df_sol['x'].iloc[i]
        dy = df_sol['y'].iloc[i + 1] - df_sol['y'].iloc[i]
        d += np.sqrt(dx * dx + dy * dy)
    return d

def plot_delta_times(sensor_times, units=1e9, title='TITLE'):
    delta_times = []
    for i in range(len(sensor_times)-1):
        dt = sensor_times[i+1]-sensor_times[i]
        delta_times.append(dt/units)
    delta_times = np.array(delta_times)
    plt.title(title)
    plt.plot(range(len(delta_times)), delta_times)
    plt.show(block=True)


def plot_relative_times(lidar_times, odo_times, title='Time vs. index (s). Relative to initial odo time'):
    time_start = odo_times[0]
    lidar_times = (lidar_times-time_start)/1e9
    odo_times = (odo_times-time_start)/1e9
    lidar_times = np.array(lidar_times)
    odo_times = np.array(odo_times)
    # lidar_times = lidar_times[0:10]
    # odo_times = odo_times[0:100]
    plt.title(title)
    plt.plot(range(len(lidar_times)), lidar_times)
    plt.plot(range(len(odo_times)), odo_times)
    plt.show(block=True)


def plot_relative_times_scatter(times1, times2, title='Time 1 vs. Time 2 (s). Relative to initial Time 1'):
    time_start = times1[0]
    times1 = np.array(times1-time_start)/1e9
    times2 = np.array(times2-time_start)/1e9
    plt.title(title)
    plt.scatter(times1, np.ones((len(times1))), label='Time 1')
    plt.scatter(times2, np.ones((len(times2))),label='Time 2')
    plt.legend()
    plt.show(block=True)



def main():
    # INDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
    # OUTDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-04-22-13-27-47'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O6-2024-04-10-11-09-24'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O7-2024-04-22-13-45-50'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O8-2024-04-24-13-05-16'
    # mixed INDOOR/OUTDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
    directory = '/media/arvc/INTENSO/DATASETS/test_arucos/test_arucos4'

    # lidarscanarray.add_lidar_scans()
    # odometry
    odoobsarray = PosesArray()
    odoobsarray.read_data(directory=directory, filename='/robot0/odom/data.csv')
    # scanmatcher
    # smobsarray = PosesArray()
    # smobsarray.read_data(directory=directory, filename='/robot0/scanmatcher/data.csv')

    # create scan Array,
    lidarscanarray = LiDARScanArray(directory=directory)
    # lidarscanarray.read_parameters()
    lidarscanarray.read_data()
    lidarscanarray.remove_orphan_lidars(pose_array=odoobsarray)


    # gpsobservations
    gpsobsarray = GPSArray()
    gpsobsarray.read_data(directory=directory, filename='/robot0/gps0/data.csv')
    gpsobsarray.read_config_ref(directory=directory)


    lidar_times = lidarscanarray.get_times()
    odo_times = odoobsarray.get_times()
    # sm_times = smobsarray.get_times()
    gps_times = gpsobsarray.get_times()

    # PLOTTING RELATIVE DELTA TIMES
    # plot_delta_times(gps_times, title='GPS delta_time (s)')
    plot_delta_times(lidar_times, title='LIDAR delta_time (s)')
    plot_delta_times(odo_times, title='ODO delta_time (s)')
    # plot_delta_times(sm_times, title='SM delta_time (s)')
    # plot_delta_times(imu_times, title='IMU delta_time (orientation, s)')

    # plot_relative_times(lidar_times, odo_times)
    plot_relative_times_scatter(times1=lidar_times, times2=odo_times, title='LIDAR-ODO')
    plot_relative_times_scatter(times1=lidar_times, times2=gps_times, title='LIDAR-GPS')

    # odo_times = np.asarray(odo_times)
    # lidar_times = np.asarray(lidar_times)
    # imu_times = np.asarray(imu_times)
    # gps_times = np.asarray(gps_times)

    print(30*'*')
    print('TOTAL MESSAGE TIMES: ')
    print('LiDAR: ', len(lidar_times))
    print('Odometry: ', len(odo_times))
    # print('GPS: ', len(gps_times))
    # print('IMU: ', len(imu_times))
    print(30*'*')

    print(30 * '*')
    print('TOTAL EPOCH TIMES (START END) ')
    print('LiDAR: ', lidar_times[0], lidar_times[-1])
    print('Odometry: ', odo_times[0], odo_times[-1])
    # print('GPS: ', gps_times[0], gps_times[-1])
    # print('IMU: ', imu_times[0], imu_times[-1])
    print(30 * '*')

    print(30 * '*')
    print('EXPERIMENT START-END (HUMAN READABLE)')
    print('LiDAR: ', datetime.fromtimestamp(lidar_times[0] // 1000000000), '/', datetime.fromtimestamp(lidar_times[-1] // 1000000000))
    print('Odometry: ', datetime.fromtimestamp(odo_times[0] // 1000000000), '/', datetime.fromtimestamp(odo_times[-1] // 1000000000))
    # print('GPS: ', datetime.fromtimestamp(gps_times[0] // 1000000000), '/', datetime.fromtimestamp(gps_times[-1] // 1000000000))
    # print('IMU: ', datetime.fromtimestamp(imu_times[0] // 1000000000), '/', datetime.fromtimestamp(imu_times[-1] // 1000000000))
    print(30 * '*')

    print(30 * '*')
    print('TOTAL EXPERIMENT TIMES (DURATION, seconds): ')
    print('LiDAR: ', (lidar_times[-1]-lidar_times[0])/1e9)
    print('Odometry: ', (odo_times[-1]-odo_times[0])/1e9)
    # print('GPS: ', (gps_times[-1]-gps_times[0])/1e9)
    # print('IMU: ', (imu_times[-1]-imu_times[0])/1e9)
    print(30 * '*')

    print(30 * '*')
    print('Sensor frequencies Hz: ')
    print('LiDAR: ', len(lidar_times)/(lidar_times[-1]-lidar_times[0])*1e9)
    print('Odometry: ', len(odo_times)/(odo_times[-1]-odo_times[0])*1e9)
    # print('GPS: ', len(gps_times)/(gps_times[-1]-gps_times[0])*1e9)
    # print('IMU: ', len(imu_times)/(imu_times[-1]-imu_times[0])*1e9)
    print(30 * '*')


if __name__ == "__main__":
    main()
