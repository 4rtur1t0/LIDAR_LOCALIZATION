from collections import deque
import numpy as np
import bisect
import matplotlib.pyplot as plt
from pyproj import Proj
from config import PARAMETERS


class GPSBuffer():
    """
    A list of observed GPS buffered observations, along with the time
    Can return:
    a) the interpolated GPS at a given time (from the two closest poses).
    b) the transformed UTM coordinates
    b) the distance two GPS.
    """
    def __init__(self, maxlen=100):
        """
        Hold a number of GPS readings. Convert to UTM when needed.
        """
        self.times = deque(maxlen=maxlen)
        self.positions = deque(maxlen=maxlen)
        self.last_processed_time = None
        self.warning_max_time_diff_s = 1
        self.config_ref = PARAMETERS.config.get('config_ref')

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        return self.positions[index]

    def get(self, index):
        return self.positions[index]

    def append(self, utm_position, time):
        self.positions.append(utm_position)
        self.times.append(time)

    def get_time(self, index):
        return self.times[index]

    def get_last_processed_time(self):
        if self.last_processed_time is None:
            return self.times[0]
        return self.last_processed_time

    def get_times(self):
        return self.times

    def get_positions(self):
        positions = np.array(self.positions)
        return positions

    def get_at_exact_time(self, timestamp):
        """
        Get the observation found at a exact, particular, time. None if not found.
        """
        index = bisect.bisect_left(self.times, timestamp)
        if index < len(self.times) and self.times[index] == timestamp:
            print(f"Element {timestamp} found at index {index}")
            return self.positions[index]
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
            return None, None
        if d1 <= d2:
            return idx1, t1
        else:
            return idx2, t2

    def closest_position_at_time(self, timestamp, delta_threshold_s=1):
        """
        Find a Position (GPS, UTM) for timestamp, the one that is closets to timestamp.
        """
        print('closes_pose_at_time!')
        # Find the index where t would be inserted in sorted_times
        idx = bisect.bisect_left(self.times, timestamp)
        print('Time distance: ', abs(timestamp-self.times[idx]))
        if abs(timestamp - self.times[idx]) > delta_threshold_s:
            print('closes_pose_at_time trying to interpolate with time difference greater than threshold')
            return None
        return self.positions[idx]

    def interpolated_gps_at_time(self, timestamp, delta_threshold_s=1):
        """
        Find a Pose for timestamp, by looking for the two closest times t1 and t2 and
        computing an interpolation
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        print('Time distances: ', (timestamp-t1), (t2-timestamp))
        if ((timestamp - t1) > delta_threshold_s) or ((t2-timestamp) > delta_threshold_s):
            print('interpolated_pose_at_time trying to interpolate with time difference greater than threshold')
            return None
        # ensures t1 < t < t2
        if t1 <= timestamp <= t2:
            gps1 = self.positions[idx1]
            gps2 = self.positions[idx2]
            p_t = self.interpolate_position(gps1, t1, gps2, t2, timestamp)
            utminterp = UTMPosition(x=p_t[0], y=p_t[1], altitude=p_t[2],
                                    covariance=gps1.position_covariance,
                                    status=gps1.status,
                                    config_ref=self.config_ref)
            return utminterp
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

    def interpolate_position(self, gps1, t1, gps2, t2, t):
        """
        Linear interpolation.
        gps1 and gps2 should be metric (UTM).
        """
        # Compute interpolation factor
        alpha = (t - t1) / (t2 - t1)
        # Linear interpolation of utm position
        p_t = (1 - alpha) * gps1.pos() + alpha * gps2.pos()
        return p_t

    def popleft(self):
        self.positions.popleft()
        self.times.popleft()



    def plot_xy(self):
        x = []
        y = []
        for i in range(len(self.times)):
            pose = self.positions[i]
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

    def plot_xy_utm(self):
        x = []
        y = []
        for i in range(len(self.times)):
            gpspos = self.positions[i]
            if gpspos.status >= 0:
                utmposition = gpspos.to_utm(config_ref=self.config_ref)
                x.append(utmposition.x)
                y.append(utmposition.y)
        plt.figure()
        plt.scatter(x, y)
        plt.show()

    def plot_xyz_utm(self):
        x = []
        y = []
        z = []
        for i in range(len(self.times)):
            gpspos = self.positions[i]
            if gpspos.status >= 0:
                utmposition = gpspos.to_utm(config_ref=self.config_ref)
                x.append(utmposition.x)
                y.append(utmposition.y)
                z.append(utmposition.altitude)
        fig = plt.figure(1)
        axes = fig.gca(projection='3d')
        plt.cla()
        axes.scatter(x, y, z, marker='o', color='red')
        plt.show()



class GPSPosition():
    def __init__(self, latitude=None, longitude=None, altitude=None, covariance=None, status=0, time=None):
        """
        Create a pose from pandas df
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.position_covariance = covariance
        self.status = status
        self.time = time

    def from_message(self, msg):
        self.latitude = msg.latitude
        self.longitude = msg.longitude
        self.altitude = msg.altitude
        self.status = msg.status.status
        self.position_covariance = np.array([msg.position_covariance[0], msg.position_covariance[4], msg.position_covariance[8]])
        self.time = msg.header.stamp.to_sec()
        return self

    def to_utm(self, config_ref):
        x, y, altitude = gps2utm(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude, config_ref=config_ref)
        return UTMPosition(x=x, y=y, altitude=altitude, covariance=self.position_covariance,
                           status=self.status,
                           config_ref=config_ref)


class UTMPosition():
    def __init__(self, x, y, altitude, covariance, status, config_ref):
        self.x = x
        self.y = y
        self.altitude = altitude
        self.position_covariance = covariance
        self.status = status
        self.config_ref = config_ref

    def pos(self):
        return np.array([self.x, self.y, self.altitude])


def gps2utm(latitude, longitude, altitude, config_ref):
    """
    Projects lat, lon to UTM coordinates
    using the origin (first lat, lon)
    """
    # latitude = lat['latitude']
    # longitude = lng['longitude']
    # altitude = altitude['altitude']
    # status = df_gps['status']

    # base reference system
    lat_ref = config_ref['latitude']
    lon_ref = config_ref['longitude']
    altitude_ref = config_ref['altitude']

    # status_array = np.array(status)
    myProj = Proj(proj='utm', zone='30', ellps='WGS84', datum='WGS84', preserve_units=False,
                  units='m')

    lat = np.array(latitude)
    lon = np.array(longitude)
    altitude = np.array(altitude)

    UTMx_ref, UTMy_ref = myProj(lon_ref, lat_ref)
    UTMx, UTMy = myProj(lon, lat)
    # UTMx = UTMx[idx]
    # UTMy = UTMy[idx]
    UTMx = UTMx - UTMx_ref
    UTMy = UTMy - UTMy_ref
    altitude = altitude - altitude_ref
    # df_gps.insert(2, 'x', UTMx, True)
    # df_gps.insert(2, 'y', UTMy, True)
    # df_gps['x'] = UTMx
    # df_gps['y'] = UTMy
    # df_gps['altitude'] = altitude
    return UTMx, UTMy, altitude
