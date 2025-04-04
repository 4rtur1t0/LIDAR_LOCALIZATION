# import numpy as np
# import yaml
import numpy as np

# from eurocreader.eurocreader import EurocReader
# import bisect
# import matplotlib.pyplot as plt
# from pyproj import Proj
# from config import PARAMETERS
import matplotlib.pyplot as plt

class Session():
    """
    A class to store observations:
    - odometry.
    - GPS
    - LiDAR pointclouds

    The class can be used to get different kind of observations (id est: GPS, lidar, odometry, imu) at different times.
    """
    def __init__(self, odo=None, smodo=None, lidar=None, aruco=None, gps=None):
        """
        Hold a number of readings.
        """
        self.odo = odo
        self.smodo = smodo # scanmatching odometry
        self.lidar = lidar
        self.aruco = aruco
        self.gps = gps
        self.times = None
        self.temp_times = None
        self.current_time = None

    def init(self):
        """
        Adding all the timestamps together, so that we can playback the experiment
        """
        global_timestamps = []
        if self.odo is not None:
            times = self.odo.get_times()
            global_timestamps.append(times)
        if self.smodo is not None:
            times = self.smodo.get_times()
            global_timestamps.append(times)
        if self.lidar is not None:
            times = self.lidar.get_times()
            global_timestamps.append(times)
        if self.aruco is not None:
            times = self.aruco.get_times()
            global_timestamps.append(times)
        if self.gps is not None:
            times = self.gps.get_times()
            global_timestamps.append(times)

        global_timestamps = np.concatenate(global_timestamps)
        global_timestamps = sorted(global_timestamps)
        self.times = global_timestamps
        self.temp_times = global_timestamps
        self.current_time = self.times[0]

    def get_next_observations(self):
        # get next time, remove from temp array
        self.current_time = self.temp_times[0]
        self.temp_times = self.temp_times[1:]
        # get the observations at these exact times
        observations = []
        odoobs = self.odo.get_at_exact_time(self.current_time)
        observations.append(('ODO', odoobs))
        smobs = self.smodo.get_at_exact_time(self.current_time)
        observations.append(('SMODO', smobs))
        arucobs, aruco_id = self.aruco.get_at_exact_time(self.current_time)
        observations.append(('ARUCO', arucobs, aruco_id))
        gpsobs = self.gps.get_at_exact_time(self.current_time)
        observations.append(('GPS', gpsobs))
        lidarobs = self.lidar.get_at_exact_time(self.current_time)
        observations.append(('LIDAR', lidarobs))
        return observations




