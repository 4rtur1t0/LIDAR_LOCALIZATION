import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.tools import slerp
from eurocreader.eurocreader import EurocReader
from artelib.vector import Vector
from artelib.quaternion import Quaternion
import bisect
import matplotlib.pyplot as plt
from collections import Counter


class PosesArray():
    """
    A list of observed poses (i. e. odometry), along with the time
    Can return:
    a) the interpolated Pose at a given time (from the two closest poses).
    b) the relative transformation T between two times.
    """
    def __init__(self, times=None, values=None):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        self.times = times
        self.values = values
        self.warning_max_time_diff_s = 1

    def __len__(self):
        return len(self.times)

    def __getitem__(self, item):
        return self.values[item]

    def read_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        df_odo = euroc_read.read_csv(filename=filename)
        self.times = df_odo['#timestamp [ns]'].to_numpy()
        self.values = []
        for index, row in df_odo.iterrows():
            self.values.append(Pose(row))

    def from_transforms(self, times, transforms):
        self.times = np.array(times)
        self.values = []
        for i in range(len(times)):
            pose = Pose(df=None)
            pose.from_transform(transforms[i])
            self.values.append(pose)

    def save_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        euroc_read.save_poses_as_csv(poses=self.values,
                                     sensor_times=self.times, filename=filename)

    def get_time(self, index):
        return self.times[index]

    def get_times(self):
        return self.times

    def get(self, index):
        return self.values[index]

    def get_poses(self):
        return self.values

    def get_transforms(self):
        transforms = []
        for i in range(len(self.times)):
            pose = self.values[i]
            T = pose.T()
            transforms.append(T)
        return transforms

    def get_at_exact_time(self, timestamp):
        """
        Get the observation found at a exact, particular, time.
        """
        index = bisect.bisect_left(self.times, timestamp)
        if index < len(self.times) and self.times[index] == timestamp:
            print(f"Element {timestamp} found at index {index}")
            return self.values[index]
        else:
            return None

    def get_closest_at_time(self, timestamp):
        d = np.abs(self.times - timestamp)
        index = np.argmin(d)
        time_diff_s = d[index] / 1e9
        output_time = self.times[index]
        output_pose = self.values[index]
        if time_diff_s > self.warning_max_time_diff_s:
            print('CAUTION!!! Found time difference (s): ', time_diff_s)
            print('CAUTION!!! Should we associate data??')
        return output_pose, output_time

    def interpolated_pose_at_time(self, timestamp, delta_threshold_s=1):
        """
        Find a Pose for timestamp, looking for the two closest times
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        print('Time distances: ', (timestamp-t1)/1e9, (t2-timestamp)/1e9)
        if ((timestamp - t1)/1e9 > delta_threshold_s) or ((t2-timestamp)/1e9 > delta_threshold_s):
            print('interpolated_pose_at_time trying to interpolate with time difference greater than threshold')
            return None
        # ensures t1 < t < t2
        if t1 <= timestamp <= t2:
            odo1 = self.values[idx1]
            odo2 = self.values[idx2]
            odointerp = self.interpolate_pose(odo1, t1, odo2, t2, timestamp)
            return odointerp
        return None

    def find_closest_times_around_t_bisect(self, t):
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.times, t)

        # Determine the two closest times
        if idx == 0:
            # t is before the first element
            return 0, self.times[0], 1, self.times[1]
        elif idx == len(self.times):
            # t is after the last element
            return -2, self.times[-2], -1, self.times[-1]
        else:
            # Take the closest two times around t
            return idx-1, self.times[idx - 1], idx,  self.times[idx]

    def interpolate_pose(self, odo1, t1, odo2, t2, t):
        # Compute interpolation factor
        alpha = (t - t1) / (t2 - t1)

        # Linear interpolation of position
        p_t = (1 - alpha) * odo1.position.pos() + alpha * odo2.position.pos()
        q1 = odo1.quaternion
        q2 = odo2.quaternion
        q_t = slerp(q1, q2, alpha)
        poset = {'x': p_t[0],
                 'y': p_t[1],
                 'z': p_t[2],
                 'qx': q_t.qx,
                 'qy': q_t.qy,
                 'qz': q_t.qz,
                 'qw': q_t.qw}
        interppose = Pose(df=poset)
        return interppose

    def plot_xy(self):
        x = []
        y = []
        for i in range(len(self.times)):
            pose = self.values[i]
            T = pose.T()
            t = T.pos()
            x.append(t[0])
            y.append(t[1])
        plt.figure()
        plt.scatter(x, y, label='odometry')
        plt.legend()
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()


class ArucoPosesArray(PosesArray):
    """
    A list of observed relative observations (observed as poses), along with the time
    and the ARUCO id observation associated to them.
    """
    def __init__(self, times=None, values=None):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        super().__init__(times, values)
        self.aruco_ids = None

    def read_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        df_odo = euroc_read.read_csv(filename=filename)
        self.times = df_odo['#timestamp [ns]'].to_numpy()
        self.aruco_ids = df_odo['aruco_id'].to_numpy()
        self.values = []
        for index, row in df_odo.iterrows():
            self.values.append(Pose(row))

    def get_aruco_ids(self):
        return self.aruco_ids

    def get_aruco_id(self, index):
        return self.aruco_ids[index]

    def get_at_exact_time(self, timestamp):
        """
        Get the observation found at a exact, particular, time.
        """
        index = bisect.bisect_left(self.times, timestamp)
        if index < len(self.times) and self.times[index] == timestamp:
            print(f"Element {timestamp} found at index {index}")
            return self.values[index], self.aruco_ids[index]
        else:
            return None

    def filter_aruco_ids(self, n_min=10):
        # first, remove ids that appear less than n_min times
        counts = Counter(self.aruco_ids)
        # then remove ids whenever the count is less than n_min
        times = []
        values = []
        aruco_ids = []
        for i in range(len(self.aruco_ids)):
            aruco_id = self.aruco_ids[i]
            # skip this observation
            if counts[aruco_id] < n_min:
                print('REMOVING SPURIOUS ARUCO_ID, observed less than', n_min)
                continue
            times.append(self.times[i])
            values.append(self.values[i])
            aruco_ids.append(self.aruco_ids[i])
        self.times = times
        self.values = values
        self.aruco_ids = aruco_ids


class ArucoLandmarksPosesArray(PosesArray):
    """
    A list of global poses of the estimated ARUCO landmarks. No time is associated to each landmarks. Considered static.
    """
    def __init__(self, times=None, values=None):
        """
        given a list of scan times (ROS times), each pcd is read on demand
        """
        super().__init__(times, values)
        self.aruco_ids = None

    def read_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        df_odo = euroc_read.read_csv(filename=filename)
        self.times = None
        self.aruco_ids = df_odo['aruco_id'].to_numpy()
        self.values = []
        for index, row in df_odo.iterrows():
            self.values.append(Pose(row))

    def get_aruco_ids(self):
        return self.aruco_ids

    def get_aruco_id(self, index):
        return self.aruco_ids[index]


class Pose():
    def __init__(self, df=None):
        """
        Create a pose from pandas df
        """
        if df is not None:
            self.position = Vector([df['x'], df['y'], df['z']])
            self.quaternion = Quaternion(qx=df['qx'], qy=df['qy'], qz=df['qz'], qw=df['qw'])
        else:
            self.position = None
            self.quaternion = None

    def from_message(self, msg):
        pos = msg.position
        ori = msg.orientation
        self.position = Vector([pos.x, pos.y, pos.z])
        self.quaternion = Quaternion(qx=ori.x, qy=ori.y, qz=ori.z, qw=ori.w)
        return self

    def from_transform(self, T):
        self.position = Vector(T.pos())
        self.quaternion = T.Q()
        return self

    def T(self):
        T = HomogeneousMatrix(self.position, self.quaternion)
        return T

    def R(self):
        return self.quaternion.R()
