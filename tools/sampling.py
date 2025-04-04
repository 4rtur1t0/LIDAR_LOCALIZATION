import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.euler import Euler
from artelib.quaternion import Quaternion
import pandas as pd


def sample_odometry(df_odo, deltaxy=0.5, deltath=0.2):
    """
    Get odometry times separated by dxy (m) and dth (rad)
    """
    df_sampled_odo = pd.DataFrame(columns=['#timestamp [ns]', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
    odo_times = []
    # sampled_indexes = []
    for ind in df_odo.index:
        # print(df_odo['x'][ind])
        position = [df_odo['x'][ind], df_odo['y'][ind], df_odo['z'][ind]]
        q = Quaternion([df_odo['qw'][ind], df_odo['qx'][ind], df_odo['qy'][ind], df_odo['qz'][ind]])
        th = q.Euler()[0]
        odo = np.array([position[0], position[1], th.abg[2]])
        current_time = df_odo['#timestamp [ns]'][ind]
        if ind == 0:
            odo_times.append(current_time)
            odoi = odo
            df_temp = pd.DataFrame(df_odo.iloc[ind])
            df_sampled_odo = pd.concat([df_sampled_odo, df_temp.T], ignore_index=True)
        odoi1 = odo

        dxy = np.linalg.norm(odoi1[0:2] - odoi[0:2])
        dth = np.linalg.norm(odoi1[2] - odoi[2])
        if dxy > deltaxy or dth > deltath:
            odo_times.append(current_time)
            odoi = odoi1
            df_temp = pd.DataFrame(df_odo.iloc[ind])
            df_sampled_odo = pd.concat([df_sampled_odo, df_temp.T], ignore_index=True)
    return np.array(odo_times), df_sampled_odo


def sample_times(sensor_times, start_index=10, delta_time=1):
    """
    Get data times separated by delta_time (s)
    """
    if delta_time is None:
        return sensor_times
    # convert to nanoseconds
    delta_time = delta_time*1*1e9
    sampled_times = []
    current_time = sensor_times[start_index]
    t = current_time
    sampled_times.append(current_time)
    for i in range(start_index, len(sensor_times)):
        current_time = sensor_times[i]
        dt = float(np.abs(current_time-t))
        if dt >= delta_time:
            sampled_times.append(current_time)
            t = current_time
    # add last time always
    sampled_times.append(current_time)
    sampled_times = np.array(sampled_times)
    # CAUTION: only unique timestamps
    sampled_times = np.unique(sampled_times)
    return np.array(sampled_times)


