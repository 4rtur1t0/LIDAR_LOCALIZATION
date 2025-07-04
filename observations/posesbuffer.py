import numpy as np
import pandas as pd
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.tools import slerp
from artelib.vector import Vector
from artelib.quaternion import Quaternion
import bisect
import matplotlib.pyplot as plt
from collections import deque


class PosesBuffer():
    """
    A list of observed poses (i. e. odometry), along with the time
    Can return:
    a) the interpolated Pose at a given time (from the two closest poses).
    b) the relative transformation T between two times.
    """
    def __init__(self, maxlen=100):
        self.times = deque(maxlen=maxlen)
        self.poses = deque(maxlen=maxlen)
        self.warning_max_time_diff_s = 1
        self.last_processed_time = None

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        return self.poses[index]

    def read_data_tum(self, directory, filename):
        full_filename = directory + filename
        df = pd.read_csv(full_filename, names=['#timestamp [ns]', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'], sep=' ')
        self.times = []  # df['#timestamp [ns]'].to_numpy()
        self.poses = []
        for index, row in df.iterrows():
            self.times.append(row['#timestamp [ns]'])
            self.poses.append(Pose(row))

    def read_data(self, directory, filename):
        """
        Caution: timestamps are stored in secs.
        """
        full_filename = directory + filename
        df = pd.read_csv(full_filename)
        for _, row in df.iterrows():
            try:
                self.times.append(float(row['#timestamp [ns]']/1e9))
            except:
                self.times.append(None)
            self.poses.append(Pose(row))

    def get(self, index):
        return self.poses[index]

    def append(self, pose, time):
        self.poses.append(pose)
        self.times.append(time)

    def get_time(self, index):
        return self.times[index]

    def get_last_processed_time(self):
        if self.last_processed_time is None:
            return self.times[0]
        return self.last_processed_time

    def get_times(self):
        return self.times

    def get_poses(self):
        return self.poses

    def get_positions(self):
        positions = []
        for i in range(len(self.times)):
            pose = self.poses[i]
            T = pose.T()
            positions.append(T.pos())
        positions = np.array(positions)
        return positions

    def get_transforms(self):
        transforms = []
        for i in range(len(self.times)):
            pose = self.poses[i]
            T = pose.T()
            transforms.append(T)
        return transforms

    def get_at_exact_time(self, timestamp):
        """
        Get the observation found at a exact, particular, time. None if not found.
        """
        index = bisect.bisect_left(self.times, timestamp)
        if index < len(self.times) and self.times[index] == timestamp:
            print(f"Element {timestamp} found at index {index}")
            return self.poses[index]
        else:
            return None

    def get_closest_index_to_time(self, timestamp, delta_threshold_s):
        """
        Get the closest pose to a given time. Warning printed if the times surpasses a threshold.
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        d1 = abs((timestamp - t1) / 1e9)
        d2 = abs((t2 - timestamp) / 1e9)
        print('Time Odo time differences times: ', d1, d2)
        if (d1 > delta_threshold_s) and (d2 > delta_threshold_s):
            print('get_index_closest_to_time could not find any close time')
            return None
        if d1 <= d2:
            return idx1 #, t1
        else:
            return idx2 #, t2


    def get_closest_pose_at_time(self, timestamp, delta_threshold_s=1.0):
        """
        Find a Pose for timestamp, the one that is closets to timestamp.
        """
        if len(self.times) == 0:
            print('get_closest_pose_at_time_fast. No poses in array.')
            return None, None
        idx = self.find_closest_index(timestamp=timestamp)
        # print('len(self.times): ', len(self.times))
        # print('idx is: ', idx)
        retrieved_time = self.times[idx]
        dt = abs(retrieved_time-timestamp)
        # print('get_closest_pose_at_time_fast. Absolute time difference: ', dt)
        if dt <= delta_threshold_s:
            # print('REturning get closest pose at time: ', self.poses[idx], self.times[idx])
            return self.poses[idx], self.times[idx]
        return None, None

    def find_closest_index(self, timestamp):
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.times, timestamp)
        if idx == 0:
            return 0
        if idx == len(self.times):
            return len(self.times)-1
        before = self.times[idx-1]
        after = self.times[idx]
        if abs(after-timestamp) < abs(timestamp-before):
            return idx
        else:
            return idx-1

    def interpolated_pose_at_time_new(self, timestamp, verbose=False):
        """
        Find a Pose for timestamp, by looking for the two closest times t1 and t2 and
        computing an interpolation
        """
        if len(self.times) == 0:
            print('No data in buffer.')
            return None, None, -1
        longitude = len(self.times)
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.times, timestamp)

        # rel_times = np.array(self.times)-self.times[0]
        # print('Rel. times:', rel_times)
        # print('Timestamp: ', timestamp-self.times[0])
        # Determine the closest times.
        # three different cases are considered, depending if the
        # t is before the first element. No interpolation performed.
        if idx == 0:
            if verbose:
                print('* FIRST ELEMENT INTERPOLATION')
            timestamp_out = self.times[0]
            interp_pose = self.poses[0]
            case_type = 0
        # t is after the last element
        elif idx == longitude:
            if verbose:
                print('** LAST ELEMENT INTERPOLATION')
            timestamp_out = self.times[-1]
            interp_pose = self.poses[-1]
            case_type = 1
        else:
            # Take the closest two times around t
            if verbose:
                print('*** BETWEEN TWO ELEMENTS INTERPOLATION')
            pose1 = self.poses[idx-1]
            pose2 = self.poses[idx]
            t1 = self.times[idx-1]
            t2 = self.times[idx]
            timestamp_out = timestamp
            interp_pose = self.interpolate_pose(pose1, t1, pose2, t2, timestamp)
            case_type = 2
        return interp_pose, timestamp_out, case_type

    def interpolated_pose_at_time(self, timestamp, delta_threshold_s=1.0):
        """
        Find a Pose for timestamp, by looking for the two closest times t1 and t2 and
        computing an interpolation
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        if idx1 is None:
            return None, None
        # print('Time distances: ', (timestamp-t1), (t2-timestamp))
        # print('Buffer length: ', len(self.times))
        # dt1 = (timestamp-t1)
        # dt2 = (t2-timestamp)
        if ((timestamp - t1) > delta_threshold_s) or ((t2-timestamp) > delta_threshold_s):
            # print('interpolated_pose_at_time trying to interpolate with time difference greater than threshold')
            return None, None
        # ensures t1 < t < t2
        if t1 <= timestamp <= t2:
            odo1 = self.poses[idx1]
            odo2 = self.poses[idx2]
            odointerp = self.interpolate_pose(odo1, t1, odo2, t2, timestamp)
            return odointerp, t1
        return None, None

    def find_closest_times_around_t_bisect(self, t):
        if len(self.times) < 2:
            print('cannot find two times in a one dim array.')
            return None, None, None, None
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
            pose = self.poses[i]
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

    def popleft(self):
        self.poses.popleft()
        self.times.popleft()


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

