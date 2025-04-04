import matplotlib.pyplot as plt
import numpy as np


def build_map(keyframes):
    map_points_global = []
    i = 0
    for keyframe in keyframes:
        print("Transforming keyframe: ", i)
        map_points_global.extend(keyframe.transform_to_global())
        i += 1
    map_points_global = np.array(map_points_global)
    return map_points_global


def plot_map(map_points, odometry=None):
    plt.figure()
    if odometry is not None:
        plt.plot(odometry[:, 0], odometry[:, 1], color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    plt.scatter(map_points[:, 0], map_points[:, 1], color='blue')
    plt.show(block=True)