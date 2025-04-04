import numpy as np
import yaml
from eurocreader.eurocreader import EurocReader
import bisect
import matplotlib.pyplot as plt
from pyproj import Proj
from config import PARAMETERS


class GPSArray():
    """
    A list of observed GPS observations, along with the time
    Can return:
    a) the interpolated GPS at a given time (from the two closest poses).
    b) the transformed UTM coordinates
    b) the distance two GPS.
    """
    def __init__(self, times=None, values=None):
        """
        Hold a number of GPS readings. Convert to UTM when needed.
        """
        self.times = times
        self.values = values
        self.warning_max_time_diff_s = 1
        self.config_ref = None

    def read_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        df_odo = euroc_read.read_csv(filename=filename)
        self.times = df_odo['#timestamp [ns]'].to_numpy()
        self.values = []
        for index, row in df_odo.iterrows():
            gpspos = GPSPosition()
            gpspos.fromdf(row)
            self.values.append(gpspos)

    def read_config_ref(self, directory):
        yaml_file_global = directory + '/' + 'robot0/gps0/reference.yaml'
        with open(yaml_file_global) as file:
            parameters = yaml.load(file, Loader=yaml.FullLoader)
        self.config_ref = parameters
        return parameters

    def filter_measurements(self, max_sigma_xy=8, min_status=0):
        max_sigma_xy=PARAMETERS.config.get('gps').get('max_sigma_xy')
        min_status = PARAMETERS.config.get('gps').get('min_status')
        filtered_values = []
        filtered_times = []
        for i in range(len(self.times)):
            gps_reading = self.values[i]
            # typically remove status -1 measurements
            if gps_reading.status < min_status:
                continue
            sigma_xy = np.sum(np.sqrt(gps_reading.covariance[0:2]))
            if sigma_xy > max_sigma_xy:
                continue
            filtered_values.append(self.values[i])
            filtered_times.append(self.times[i])
        self.values = np.array(filtered_values)
        self.times = np.array(filtered_times)

    def save_data(self, directory, filename):
        euroc_read = EurocReader(directory=directory)
        euroc_read.save_transforms_as_csv(transforms=self.values,
                                          sensor_times=self.times, filename=filename)

    def get_time(self, index):
        return self.times[index]

    def get_times(self):
        return self.times

    def get(self, index):
        return self.values[index]

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

    def get_utm_positions(self):
        utmpositions = []
        for gpsvalue in self.values:
            utmpositions.append(gpsvalue.to_utm(config_ref=self.config_ref))
        return utmpositions

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

    def interpolated_utm_at_time(self, timestamp, delta_threshold_s=1):
        """
        Find a Pose for timestamp, looking for the two closest times
        Every GPS reading is converted to UTM considering the GPS reference.
        """
        idx1, t1, idx2, t2 = self.find_closest_times_around_t_bisect(timestamp)
        print('Time distances: ', (timestamp-t1)/1e9, (t2-timestamp)/1e9)
        # ensure t1 < t < t2 and the time distances are below a threshold s
        if ((timestamp - t1)/1e9 > delta_threshold_s) or ((t2-timestamp)/1e9 > delta_threshold_s):
            print('Discard GPS!')
            return None
        if t1 <= timestamp <= t2:
            gps1 = self.values[idx1]
            gps2 = self.values[idx2]
            utm1 = gps1.to_utm(config_ref=self.config_ref)
            utm2 = gps2.to_utm(config_ref=self.config_ref)
            odointerp = self.interpolate_utm(utm1, t1, utm2, t2, timestamp)
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

    def interpolate_utm(self, utm1, t1, utm2, t2, t):
        # Compute interpolation factor
        alpha = (t - t1) / (t2 - t1)
        # Linear interpolation of position and altitude
        p_t = (1 - alpha) * utm1.pos() + alpha * utm2.pos()
        status = min(utm1.status, utm2.status)
        # covariance as the maximum of the two
        interpcovariance = np.maximum(utm1.covariance, utm2.covariance)
        interputm = UTMPosition(x=p_t[0], y=p_t[1], altitude=p_t[2],
                                covariance=interpcovariance,
                                status=status)
        return interputm

    def plot_xy_utm(self):
        x = []
        y = []
        for i in range(len(self.times)):
            gpspos = self.values[i]
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
            gpspos = self.values[i]
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
    def __init__(self, latitude=None, longitude=None, altitude=None, covariance=None, status=0):
        """
        Create a pose from pandas df
        """
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.covariance = covariance
        self.status = status

    def fromdf(self, df):
        """
        Create a pose from pandas df
        """
        self.latitude = df['latitude']
        self.longitude = df['longitude']
        self.altitude = df['altitude']
        self.covariance = np.array([df['covariance_d1'], df['covariance_d2'], df['covariance_d3']])
        self.status = df['status']

    def to_utm(self, config_ref):
        x, y, altitude = gps2utm(latitude=self.latitude, longitude=self.longitude, altitude=self.altitude, config_ref=config_ref)
        return UTMPosition(x=x, y=y, altitude=altitude, covariance=self.covariance, status=self.status)


class UTMPosition():
    def __init__(self, x, y, altitude, covariance, status):
        self.x = x
        self.y = y
        self.altitude = altitude
        self.covariance = covariance
        self.status = status

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
