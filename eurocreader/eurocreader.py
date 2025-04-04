import numpy as np
import yaml
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.quaternion import Quaternion
import pandas as pd
import os


class EurocReader():
    def __init__(self, directory):
        self.directory = directory

    def read_transform(self, sensor):
        print('Reading transformation for sensor: ', sensor)
        transform_yaml_filename = self.directory + '/robot0/' + sensor + '/transform.yaml'
        print('Transformation file is: ', transform_yaml_filename)
        with open(transform_yaml_filename) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        T = HomogeneousMatrix(config['transform'])
        print('Found transformation: ')
        T.print_nice()
        return T

    def read_utm_ref(self, gpsname):
        print('Reading base latitude/longitude/altitude: ')
        reference_yaml_filename = self.directory + '/robot0/' + gpsname + '/reference.yaml'
        print('Transformation file is: ', reference_yaml_filename)
        with open(reference_yaml_filename) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        lat = config['latitude']
        lon = config['longitude']
        altitude = config['altitude']
        print('Found reference (lag/lon/altitude): ', lat, lon, altitude)
        return config

    def read_csv(self, filename):
        full_filename = self.directory + filename
        df = pd.read_csv(full_filename)
        return df

    def save_csv(self, df, filename):
        df.to_csv(self.directory+filename)

    def save_transforms_as_csv(self, sensor_times, transforms, filename):
        global_filename = self.directory+filename
        global_directory = os.path.dirname(os.path.abspath(global_filename))
        try:
            os.makedirs(global_directory)
        except OSError:
            print("Directory exists or creation failed", global_directory)
        data_list = []
        for i in range(len(transforms)):
            Ti = transforms[i]
            t = Ti.pos()
            q = Ti.Q()
            data_list.append({'timestamp': sensor_times[i],
                              'x': t[0], 'y': t[1], 'z': t[2],
                              'qx': q.qx, 'qy': q.qy, 'qz': q.qz, 'qw': q.qw})
        df = pd.DataFrame(data_list, columns=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        df.to_csv(self.directory+filename, index=False, header=['#timestamp [ns]',
                                   'x', 'y', 'z',
                                   'qx', 'qy', 'qz', 'qw'])
        return df

    def save_poses_as_csv(self, sensor_times, poses, filename):
        global_filename = self.directory+filename
        global_directory = os.path.dirname(os.path.abspath(global_filename))
        try:
            os.makedirs(global_directory)
        except OSError:
            print("Directory exists or creation failed", global_directory)
        data_list = []
        for i in range(len(poses)):
            pi = poses[i]
            t = pi.position
            q = pi.quaternion
            data_list.append({'timestamp': sensor_times[i],
                              'x': t[0], 'y': t[1], 'z': t[2],
                              'qx': q.qx, 'qy': q.qy, 'qz': q.qz, 'qw': q.qw})
        df = pd.DataFrame(data_list, columns=['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        df.to_csv(self.directory+filename, index=False, header=['#timestamp [ns]',
                                   'x', 'y', 'z',
                                   'qx', 'qy', 'qz', 'qw'])
        return df

    def save_sensor_times_as_csv(self, sensor_times, filename):
        global_filename = self.directory+filename
        global_directory = os.path.dirname(os.path.abspath(global_filename))
        try:
            os.makedirs(global_directory)
        except OSError:
            print("Directory exists or creation failed", global_directory)
        data_list = []
        for i in range(len(sensor_times)):
            data_list.append({'#timestamp [ns]': sensor_times[i]})
        df = pd.DataFrame(data_list)
        df.to_csv(self.directory+filename)
        return df

    def save_loop_closures_as_csv(self, loop_closures, filename):
        global_filename = self.directory+filename
        global_directory = os.path.dirname(os.path.abspath(global_filename))
        try:
            os.makedirs(global_directory)
        except OSError:
            print("Directory exists or creation failed", global_directory)
        data_list = []
        for lc in loop_closures:
            if lc is None:
                continue
            for pair in lc:
                data_list.append({'i': pair[0], 'j': pair[1]})
        df = pd.DataFrame(data_list)
        df.to_csv(global_filename)
        return df

    def save_landmarks_as_csv(self, landmark_ids, transforms, filename):
        global_filename = self.directory+filename
        global_directory = os.path.dirname(os.path.abspath(global_filename))
        try:
            os.makedirs(global_directory)
        except OSError:
            print("Directory exists or creation failed", global_directory)
        data_list = []
        for i in range(len(transforms)):
            Ti = transforms[i]
            t = Ti.pos()
            q = Ti.Q()
            data_list.append({'aruco_id': landmark_ids[i],
                              'x': t[0], 'y': t[1], 'z': t[2],
                              'qx': q.qx, 'qy': q.qy, 'qz': q.qz, 'qw': q.qw})
        df = pd.DataFrame(data_list, columns=['aruco_id', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        df.to_csv(self.directory+filename, index=False, header=['aruco_id',
                                   'x', 'y', 'z',
                                   'qx', 'qy', 'qz', 'qw'])
        return df

    # def read_scan_times_numpy(self, directory='/robot0/lidar/data.csv'):
    #     scan_times_csv_filename = self.directory + directory
    #     df_scan_times = pd.read_csv(scan_times_csv_filename)
    #     return df_scan_times

    def sample_odometry(self, deltaxy=0.5, deltath=0.2):
        """
        Get odometry times separated by dxy (m) and dth (rad)
        """
        df_odo = self.read_csv('/robot0/lidar/data.csv')
        odo_times = []
        for ind in df_odo.index:
            # print(df_odo['x'][ind])
            position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
            q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
            th = q.Euler()
            odo = np.array([position[0], position[1], th.abg[2]])
            current_time = df_odo['#timestamp [ns]'][ind]
            if ind == 0:
                odo_times.append(current_time)
                odoi = odo
            odoi1 = odo
            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])
            if dxy > deltaxy or dth > deltath:
                odo_times.append(current_time)
                odoi = odoi1
        return np.array(odo_times)

    def get_closest_times(self, master_sensor_times, sensor_times, warning_max_time_dif_s=0.5*1e9):
        """
        For each time in master_sensor_times, find the closest time in sensor_times.
        The resulting time vector has the same dimensions as master_sensor_times
        """
        output_times = []
        # for each master_sensor_times, find the closest time in sensor_times
        for timestamp in master_sensor_times:
            d = np.abs(sensor_times-timestamp)
            index = np.argmin(d)
            time_diff_s = d[index]
            output_times.append(sensor_times[index])
            if time_diff_s > warning_max_time_dif_s:
                print('CAUTION!!! Found time difference (s): ', time_diff_s/1e9)
                print('CAUTION!!! Should we associate data??')
        output_times = np.array(output_times)
        return output_times

    def get_df_at_times(self, df_data, time_list):
        """
        Build a pandas df from exaclty the times specified
        """
        df = pd.DataFrame(columns=['#timestamp [ns]', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        # now find odo corresponding to closest times
        for timestamp in time_list:
            # find the exact timestamp in df
            ind = df_data['#timestamp [ns]'] == timestamp
            row = df_data.loc[ind]
            df = pd.concat([df, row], ignore_index=True)
        return df

