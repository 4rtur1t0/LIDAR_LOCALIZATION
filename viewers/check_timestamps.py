"""
Visualize delta_t for each sensor.
"""
from eurocreader.eurocreader import EurocReader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


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


def main():
    # INDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I1-2024-03-06-13-44-09'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I2-2024-03-06-13-50-58'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR/I3-2024-04-22-15-21-28'
    # OUTDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O1-2024-03-06-17-30-39'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O2-2024-03-07-13-33-34'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O3-2024-03-18-17-11-17'
    directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O4-2024-04-22-13-27-47'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O5-2024-04-24-12-47-35'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O6-2024-04-10-11-09-24'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O7-2024-04-22-13-45-50'
    # directory = '/media/arvc/INTENSO/DATASETS/OUTDOOR/O8-2024-04-24-13-05-16'
    # mixed INDOOR/OUTDOOR
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
    # directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO1-2024-05-03-09-51-52'
    directory = '/media/arvc/INTENSO/DATASETS/test_arucos/test_arucos4'

    euroc_read = EurocReader(directory=directory)

    # read LiDAR times
    df_lidar = euroc_read.read_csv(filename='/robot0/lidar/data.csv')
    lidar_times = df_lidar['#timestamp [ns]']

    # read odometry times
    df_odo = euroc_read.read_csv(filename='/robot0/odom/data.csv')
    odo_times = df_odo['#timestamp [ns]']
    print('ODOMETRY TRAVELLED DISTANCE: ', computed_distance_travelled(df_odo))


    # read gps times
    try:
        df_gps = euroc_read.read_csv(filename='/robot0/gps0/data.csv')
        gps_times = df_gps['#timestamp [ns]']
    except FileNotFoundError:
        print('No GPS data found')
        gps_times = [0, 0]

    # read IMU times
    try:
        df_imu = euroc_read.read_csv(filename='/robot0/imu0/orientation/data.csv')
        imu_times = df_imu['#timestamp [ns]']
    except FileNotFoundError:
        print('No IMU data found')
        imu_times = [0, 0]

    # PLOTTING RELATIVE DELTA TIMES
    plot_delta_times(gps_times, title='GPS delta_time (s)')
    plot_delta_times(lidar_times, title='LIDAR delta_time (s)')
    plot_delta_times(odo_times, title='ODO delta_time (s)')
    plot_delta_times(imu_times, title='IMU delta_time (orientation, s)')

    odo_times = np.asarray(odo_times)
    lidar_times = np.asarray(lidar_times)
    imu_times = np.asarray(imu_times)
    gps_times = np.asarray(gps_times)

    print(30*'*')
    print('TOTAL MESSAGE TIMES: ')
    print('LiDAR: ', len(lidar_times))
    print('Odometry: ', len(odo_times))
    print('GPS: ', len(gps_times))
    print('IMU: ', len(imu_times))
    print(30*'*')

    print(30 * '*')
    print('TOTAL EPOCH TIMES (START END) ')
    print('LiDAR: ', lidar_times[0], lidar_times[-1])
    print('Odometry: ', odo_times[0], odo_times[-1])
    print('GPS: ', gps_times[0], gps_times[-1])
    print('IMU: ', imu_times[0], imu_times[-1])
    print(30 * '*')

    print(30 * '*')
    print('EXPERIMENT START-END (HUMAN READABLE)')
    print('LiDAR: ', datetime.fromtimestamp(lidar_times[0] // 1000000000), '/', datetime.fromtimestamp(lidar_times[-1] // 1000000000))
    print('Odometry: ', datetime.fromtimestamp(odo_times[0] // 1000000000), '/', datetime.fromtimestamp(odo_times[-1] // 1000000000))
    print('GPS: ', datetime.fromtimestamp(gps_times[0] // 1000000000), '/', datetime.fromtimestamp(gps_times[-1] // 1000000000))
    print('IMU: ', datetime.fromtimestamp(imu_times[0] // 1000000000), '/', datetime.fromtimestamp(imu_times[-1] // 1000000000))
    print(30 * '*')

    print(30 * '*')
    print('TOTAL EXPERIMENT TIMES (DURATION, seconds): ')
    print('LiDAR: ', (lidar_times[-1]-lidar_times[0])/1e9)
    print('Odometry: ', (odo_times[-1]-odo_times[0])/1e9)
    print('GPS: ', (gps_times[-1]-gps_times[0])/1e9)
    print('IMU: ', (imu_times[-1]-imu_times[0])/1e9)
    print(30 * '*')

    print(30 * '*')
    print('Sensor frequencies Hz: ')
    print('LiDAR: ', len(lidar_times)/(lidar_times[-1]-lidar_times[0])*1e9)
    print('Odometry: ', len(odo_times)/(odo_times[-1]-odo_times[0])*1e9)
    print('GPS: ', len(gps_times)/(gps_times[-1]-gps_times[0])*1e9)
    print('IMU: ', len(imu_times)/(imu_times[-1]-imu_times[0])*1e9)
    print(30 * '*')


if __name__ == "__main__":
    main()
