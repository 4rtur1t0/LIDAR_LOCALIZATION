import matplotlib.pyplot as plt
import numpy as np
from artelib.quaternion import Quaternion
import tilemapbase
import pandas as pd

def plot(x, title='Untitled', block=True):
    plt.figure()
    x = np.array(x)
    plt.plot(x)
    plt.title(title)
    plt.show(block=block)


def plot_vars(q_path, title='UNTITLED'):
    plt.figure()
    q_path = np.array(q_path)
    sh = q_path.shape
    for i in range(0, sh[1]):
        plt.plot(q_path[:, i], label='q' + str(i + 1), linewidth=4)
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot_xy(x, y, title='UNTITLED'):
    plt.figure()
    x = np.array(x)
    y = np.array(y)
    x = x.flatten()
    y = y.flatten()
    plt.plot(x, y, linewidth=4, marker='.')
    # plt.xlabel('qi')
    # plt.ylabel('dw')
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot3d(x, y, z, title='3D'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.xlim([0, 0.5])
    plt.title(title)
    plt.show(block=True)


def plot_path(odo_icp, odo_gt, title='UNTITLED'):
    plt.figure()
    plt.plot(odo_icp[:, 0], odo_icp[:, 1], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.plot(odo_gt[:, 0], odo_gt[:, 1], color='red', marker='o', markerfacecolor='blue', markersize=12)
    plt.legend()
    plt.title(title)
    plt.show(block=True)


def plot_state(x_gt, odo, title='UNTITLED'):
    plt.figure()
    plt.plot(odo[:, 0], odo[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.title(title)
    plt.show(block=False)


def plot_initial(x_gt, odo, edges, title='UNTITLED'):
    plt.figure()
    plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.plot(odo[:, 0], odo[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    for edge in edges:
        i = int(edge[0])
        j = int(edge[1])
        x = [odo[i, 0], odo[j, 0]]
        y = [odo[i, 1], odo[j, 1]]
        plt.plot(x, y, color='black', linestyle='dotted', marker='o', markerfacecolor='black', markersize=12)
    plt.title(title)
    plt.show(block=False)


def plot_x(x, title='UNTITLED'):
    N = int(len(x)/3)
    sol = np.zeros((N, 3))
    for i in range(N):
        sol[i][0]=x[i*3]
        sol[i][1] = x[i * 3 + 1]
        sol[i][2] = x[i * 3 + 2]
    plt.figure()
    plt.plot(sol[:, 0], sol[:, 1], color='blue', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=12)
    # plt.plot(x_gt[:, 0], x_gt[:, 1], color='red', marker='o', markerfacecolor='red', markersize=12)
    plt.title(title)
    plt.show(block=False)


def plot_quaternion_data(df_data, title='Quaternion data to Euler', annotate_time=False):
    eul = []
    timestamps = []
    for i in range(len(df_data)):
        qx = df_data['qx'][i]
        qy = df_data['qy'][i]
        qz = df_data['qz'][i]
        qw = df_data['qw'][i]
        q = [qw, qx, qy, qz]
        Q = Quaternion(q)
        th = Q.Euler()[0]
        th.abg = th.abg + np.array([0, 0, -2.5])
        th.abg[2] = np.arctan2(np.sin(th.abg[2]), np.cos(th.abg[2]))
        eul.append(th.abg)
        timestamps.append(df_data['#timestamp [ns]'][i]/1e9)
    eul = np.array(eul)
    timestamps = np.array(timestamps)-df_data['#timestamp [ns]'][0]/1e9

    plt.figure()
    plt.title(title)
    if annotate_time:
        plt.plot(timestamps, eul[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
        plt.plot(timestamps, eul[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
        plt.plot(timestamps, eul[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
        plt.xlabel('Experiment time (s)')
    else:
        plt.plot(range(len(eul)), eul[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
        plt.plot(range(len(eul)), eul[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
        plt.plot(range(len(eul)), eul[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
        plt.xlabel('Data index')
    plt.show(block=True)


def plot_xyz_data(df_data, title='TITLE', annotate_time=False):
    plt.figure()
    plt.title(title)
    if annotate_time:
        timestamps = df_data['#timestamp [ns]'].to_numpy()/1e9
        timestamps = timestamps - df_data['#timestamp [ns]'][0] / 1e9
        plt.plot(timestamps, df_data['x'], color='red', linestyle='dashed', marker='o', markersize=12)
        plt.plot(timestamps, df_data['y'], color='green', linestyle='dashed', marker='o', markersize=12)
        plt.plot(timestamps, df_data['z'], color='blue', linestyle='dashed', marker='o', markersize=12)
    else:
        plt.plot(range(len(df_data)), df_data['x'], color='red', linestyle='dashed', marker='o', markersize=12)
        plt.plot(range(len(df_data)), df_data['y'], color='green', linestyle='dashed', marker='o', markersize=12)
        plt.plot(range(len(df_data)), df_data['z'], color='blue', linestyle='dashed', marker='o', markersize=12)
    plt.show()


def plot_xy_data(df_data, title='TITLE', sample=10, annotate_time=False, annotate_index=False):
    plt.figure()
    plt.title(title)
    plt.scatter(df_data['x'], df_data['y'], color='blue')
    if annotate_time:
        for i in range(0, len(df_data['x']), sample):
            exp_time = (df_data['#timestamp [ns]'][i]-df_data['#timestamp [ns]'][0])/1e9
            txt = "{:.3f}".format(exp_time)
            plt.annotate(txt, (df_data['x'][i], df_data['y'][i]), fontsize=12)
    if annotate_index:
        for i in range(0, len(df_data), 10):
            txt = str(i)
            plt.annotate(txt, (df_data['x'].iloc[i], df_data['y'].iloc[i]), fontsize=12)
    plt.show()

def plot_3D_data(df_data):
    """
        Print and plot the result simply. in 3D
    """

    fig = plt.figure(0)
    axes = fig.gca(projection='3d')
    plt.cla()
    axes.scatter(df_data['x'], df_data['y'], df_data['z'])





def compute_distance(lat1, lng1, lat2, lng2):
    from math import sin, cos, sqrt, atan2, radians
    # approximate radius of earth in m
    R = 6373.0 * 1000.0
    lat1 = radians(lat1)
    lng1 = radians(lng1)
    lat2 = radians(lat2)
    lng2 = radians(lng2)

    dlon = lng2 - lng1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

def plot_gps_points(df_gps, title='GPS POINTS (index, radius error in meters)',
                    annotate_index=False, annotate_error=False):
    """
    df_gt_gps: ground truth
    df_trajectory_gps: the trajectory
    :param df_gt_gps:
    :param zoom:
    :param color:
    :return:
    """
    plt.figure()
    plt.scatter(x=df_gps['longitude'], y=df_gps['latitude'])
    plt.title(title)
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    if annotate_index:
        for i in range(0, len(df_gps), 10):
            txt = str(i)
            # print(i, df_gps['longitude'].iloc[i], df_gps['latitude'].iloc[i])
            plt.annotate(txt, (df_gps['longitude'].iloc[i], df_gps['latitude'].iloc[i]), fontsize=12)
    if annotate_error:
        for i in range(0, len(df_gps['longitude']), 10):
            s_x = 2 * np.sqrt(df_gps['covariance_d1'].iloc[i])
            txt = "{:.3f}".format(s_x)
            plt.annotate(txt, (df_gps['longitude'].iloc[i], df_gps['latitude'].iloc[i]), fontsize=12)
    plt.show(block=True)

def plot_utm_points(df_gps, title='UTM POINTS (index, radius error in meters)',
                    annotate_index=False, annotate_error=False):
    """
    df_gt_gps: ground truth
    df_trajectory_gps: the trajectory
    :param df_gt_gps:
    :param zoom:
    :param color:
    :return:
    """
    plt.figure()
    plt.scatter(x=df_gps['x'], y=df_gps['y'])
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    if annotate_index:
        for i in range(0, len(df_gps['x']), 10):
            txt = str(i)
            plt.annotate(txt, (df_gps['x'].iloc[i], df_gps['y'].iloc[i]), fontsize=12)
    if annotate_error:
        for i in range(0, len(df_gps['x']), 10):
            s_x = 2 * np.sqrt(df_gps['covariance_d1'].iloc[i])
            txt = "{:.3f}".format(s_x)
            plt.annotate(txt, (df_gps['x'].iloc[i], df_gps['y'].iloc[i]), fontsize=12)
    plt.show(block=True)


def plot_gps_OSM(df_gps, expand=0.001, save_fig=False):
    tilemapbase.init(create=True)

    extent = tilemapbase.Extent.from_lonlat(
        df_gps.longitude.min() - expand,
        df_gps.longitude.max() + expand,
        df_gps.latitude.min() - expand,
        df_gps.latitude.max() + expand,
        )
    trip_projected = df_gps.apply(
        lambda x: tilemapbase.project(x.longitude, x.latitude), axis=1
    ).apply(pd.Series)
    trip_projected.columns = ["x", "y"]

    tiles = tilemapbase.tiles.build_OSM()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    plotter = tilemapbase.Plotter(extent, tiles, height=600)
    plotter.plot(ax, tiles, alpha=0.8)
    ax.plot(trip_projected.x, trip_projected.y, color='blue', linewidth=1)
    # ax.scatter(trip_projected.x, trip_projected.y, color='blue')
    plt.axis('off')
    plt.show(block=True)
    if save_fig:
        fig.savefig('trip.png', bbox_inches='tight', pad_inches=0, dpi=300)



#
# def error_ellipse(ax, xc, yc, cov, sigma=10, **kwargs):
#     """
#     Plot an error ellipse contour over your data.
#     Inputs:
#     ax : matplotlib Axes() object
#     xc : x-coordinate of ellipse center
#     yc : x-coordinate of ellipse center
#     cov : covariance matrix
#     sigma : # sigma to plot (default 1)
#     additional kwargs passed to matplotlib.patches.Ellipse()
#     """
#     w, v = np.linalg.eigh(cov) # assumes symmetric matrix
#     order = w.argsort()[::-1]
#     w, v = w[order], v[:, order]
#     theta = np.degrees(np.arctan2(*v[:, 0][::-1]))
#     ellipse = Ellipse(xy=(xc, yc),
#                       width=2.*sigma*np.sqrt(w[0]),
#                       height=2.*sigma*np.sqrt(w[1]),
#                       angle=theta, **kwargs)
#     ellipse.set_facecolor('none')
#     ax.add_patch(ellipse)
#
#
# def plot_gps_points_w_covariance(df_gps, title='GPS POINTS', sample=10):
#     """
#     df_gt_gps: ground truth
#     df_trajectory_gps: the trajectory
#     :param df_gt_gps:
#     :param zoom:
#     :param color:
#     :return:
#     """
#     CUIDADO: SE DEBE PLOTEAR, LÓGICAMENTE EN COORDENADAS UTM, PARA PODER PONER ELIPSES DE ERROR EN M
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(x=df_gps['longitude'], y=df_gps['latitude'])
#     # ax.set_title(title)
#     # ax.title(title)
#     # plt.setxlabel('longitude')
#     # plt.ylabel('latitude')
#     for i in range(0, len(df_gps['longitude']), sample):
#         xi = df_gps['longitude'][i]
#         yi = df_gps['latitude'][i]
#         # ojo, covariances are in meters.
#         cov_x = df_gps['covariance_d1'][i]
#         cov_y = df_gps['covariance_d2'][i]
#         cov = np.array([[cov_x, 0], [0, cov_y]])
#         error_ellipse(ax, xi, yi, cov, ec='red')
#         # plt.annotate(txt, (df_gps['longitude'][i], df_gps['latitude'][i]), fontsize=12)
#
#     plt.show(block=True)


#     plt.plot(range(len(data)), data[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
    #     plt.plot(range(len(data)), data[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
    #     plt.plot(range(len(data)), data[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
    #     plt.show(block=True)
# def eval_errors(ground_truth_transforms, measured_transforms):
#     # compute xyz alpha beta gamma
#     gt_tijs = []
#     meas_tijs = []
#     for i in range(len(ground_truth_transforms)):
#         gt_tijs.append(ground_truth_transforms[i].t2v(n=3))  # !!! convert to x y z alpha beta gamma
#         meas_tijs.append(measured_transforms[i].t2v(n=3))
#
#     gt_tijs = np.array(gt_tijs)
#     meas_tijs = np.array(meas_tijs)
#     errors = gt_tijs-meas_tijs
#
#     plt.figure()
#     plt.plot(range(len(errors)), errors[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(errors)), errors[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(errors)), errors[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
#     plt.title('Errors XYZ')
#     plt.show(block=True)
#
#     plt.figure()
#     plt.plot(range(len(errors)), errors[:, 3], color='red', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(errors)), errors[:, 4], color='green', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(errors)), errors[:, 5], color='blue', linestyle='dashed', marker='o', markersize=12)
#     plt.title('Errors Alfa Beta Gamma')
#     plt.show(block=True)
#
#     print("Covariance matrix: ")
#     print(np.cov(errors.T))
#
#
# def view_pos_data(data):
#     plt.figure()
#     plt.plot(range(len(data)), data[:, 0], color='red', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(data)), data[:, 1], color='green', linestyle='dashed', marker='o', markersize=12)
#     plt.plot(range(len(data)), data[:, 2], color='blue', linestyle='dashed', marker='o', markersize=12)
#     plt.show(block=True)
#
#     plt.figure()
#     plt.plot(data[:, 0], data[:, 1], color='blue', linestyle='dashed', marker='o', markersize=12)
#     plt.show(block=True)
#
#


# def plot_delta_times(sensor_times, units=1e9):
#     delta_times = []
#     for i in range(len(sensor_times)-1):
#         dt = sensor_times[i+1]-sensor_times[i]
#         delta_times.append(dt/units)
#     delta_times = np.array(delta_times)
#     plt.plot(range(len(delta_times)), delta_times)
#     plt.show()
