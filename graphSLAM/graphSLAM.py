"""

"""
from __future__ import print_function
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from gtsam.symbol_shorthand import X, L
from eurocreader.eurocreader import EurocReader

# Declare the 3D translational standard deviations of the prior factor's Gaussian model, in meters.
prior_xyz_sigma = 1000.0000000
# Declare the 3D rotational standard deviations of the prior factor's Gaussian model, in degrees.
prior_rpy_sigma = 100.0000000
# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
odo_xyz_sigma = 0.1
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
odo_rpy_sigma = 5
# Declare the 3D translational standard deviations of the scanmatcher factor's Gaussian model, in meters.
icp_xyz_sigma = 0.01
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
icp_rpy_sigma = 2
# GPS noise: in UTM, x, y, height. Standar errors. GPS noise can be included from the GPS readings.
# gps_xy_sigma = 2.5
# gps_altitude_sigma = 3.0
gps_xy_sigma = 8
gps_altitude_sigma = 100.0

# Declare the 3D translational standard deviations of the odometry factor's Gaussian model, in meters.
aruco_xyz_sigma = 0.05
# Declare the 3D rotational standard deviations of the odometry factor's Gaussian model, in degrees.
aruco_rpy_sigma = 5

# Declare the noise models
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_rpy_sigma*np.pi/180,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma,
                                                         prior_xyz_sigma]))
# noise from the scanmatcher
SM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([icp_rpy_sigma*np.pi/180,
                                                            icp_rpy_sigma*np.pi/180,
                                                            icp_rpy_sigma*np.pi/180,
                                                            icp_xyz_sigma,
                                                            icp_xyz_sigma,
                                                            icp_xyz_sigma]))
# the noise is twice as big
MAPSM_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([15*icp_rpy_sigma*np.pi/180,
                                                            15*icp_rpy_sigma*np.pi/180,
                                                            15*icp_rpy_sigma*np.pi/180,
                                                            50*icp_xyz_sigma,
                                                            50*icp_xyz_sigma,
                                                            50*icp_xyz_sigma]))

ODO_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([odo_rpy_sigma*np.pi/180,
                                                            odo_rpy_sigma*np.pi/180,
                                                            odo_rpy_sigma*np.pi/180,
                                                            odo_xyz_sigma,
                                                            odo_xyz_sigma,
                                                            odo_xyz_sigma]))

GPS_NOISE = gtsam.Point3(gps_xy_sigma, gps_xy_sigma, gps_altitude_sigma)

ARUCO_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([aruco_rpy_sigma*np.pi/180,
                                                            aruco_rpy_sigma*np.pi/180,
                                                            aruco_rpy_sigma*np.pi/180,
                                                            aruco_xyz_sigma,
                                                            aruco_xyz_sigma,
                                                            aruco_xyz_sigma]))

class GraphSLAM():
    def __init__(self, T0, Tlidar_gps, Tlidar_cam, max_number_of_landmarks=1000):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()
        # transforms
        self.T0 = T0
        self.Tlidar_gps = Tlidar_gps
        self.Tgps_lidar = Tlidar_gps.inv()
        self.Tlidar_cam = Tlidar_cam
        # noises
        self.PRIOR_NOISE = PRIOR_NOISE
        self.SM_NOISE = SM_NOISE
        self.MAPSM_NOISE = MAPSM_NOISE
        self.ODO_NOISE = ODO_NOISE
        self.GPS_NOISE = gtsam.noiseModel.Diagonal.Sigmas(GPS_NOISE)
        # landmarks
        self.max_number_of_landmarks = max_number_of_landmarks
        # Solver parameters
        parameters = gtsam.ISAM2Params()
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(parameters)
        # self.skip_optimization = skip_optimization

    def init_graph(self):
        T = self.T0
        # init graph starting at 0 and with initial pose T0 = eye
        self.graph.push_back(gtsam.PriorFactorPose3(X(0), gtsam.Pose3(T.array), self.PRIOR_NOISE))
        # CAUTION: the initial T0 transform is the identity.
        self.initial_estimate.insert(X(0), gtsam.Pose3(T.array))
        self.current_estimate.insert(X(0), gtsam.Pose3(T.array))

    def check_estimate(self, i):
        print("X(i) exists?: ", i)
        print(self.current_estimate.exists(X(i)))
        if self.current_estimate.exists(X(i)):
            print(self.current_estimate.atPose3(X(i)))
            return True
        return False

    def add_initial_estimate(self, atb, k):
        next_estimate = self.current_estimate.atPose3(X(k-1)).compose(gtsam.Pose3(atb.array))
        while True:
            self.initial_estimate.insert(X(k), next_estimate)
            if self.initial_estimate.exists(X(k)):
                break
            print('Retrying insert initial_estimate. Next estimate is:')
            print(next_estimate)
        while True:
            self.current_estimate.insert(X(k), next_estimate)
            if self.current_estimate.exists(X(k)):
                break
            print('Retrying insert current_estimate. Next estimate is:')
            print(next_estimate)




    # def add_initial_landmark_estimate(self, atb, k, landmark_id):
    #     """
    #     Landmark k observed from pose i
    #     """
    #     landmark_estimate = self.current_estimate.atPose3(X(k)).compose(gtsam.Pose3(atb.array))
    #     self.initial_estimate.insert(L(landmark_id), landmark_estimate)
    #     self.current_estimate.insert(L(landmark_id), landmark_estimate)

    def add_edge(self, atb, i, j, noise_type):
        """
        Adds edge between poses i and j
        """
        noise = self.select_noise(noise_type)
        # add consecutive observation
        self.graph.push_back(gtsam.BetweenFactorPose3(X(i), X(j), gtsam.Pose3(atb.array), noise))

    def add_edge_pose_landmark(self, atb, i, j, sigmas):
        """
        Adds edge between poses i and j
        """
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigmas[0]*np.pi/180,
                                                            sigmas[1]*np.pi/180,
                                                            sigmas[2]*np.pi/180,
                                                            sigmas[3],
                                                            sigmas[4],
                                                            sigmas[5]]))
        # add consecutive observation
        self.graph.push_back(gtsam.BetweenFactorPose3(X(i), L(j), gtsam.Pose3(atb.array), noise))

    def add_GPSfactor(self, utmx, utmy, utmaltitude, gpsnoise, i):
        utm = gtsam.Point3(utmx, utmy, utmaltitude)
        if gpsnoise is None:
            self.graph.add(gtsam.GPSFactor(X(i), utm, self.GPS_NOISE))
        else:
            gpsnoise = gtsam.Point3(gpsnoise[0], gpsnoise[1], gpsnoise[2])
            gpsnoise = gtsam.noiseModel.Diagonal.Sigmas(sigmas=gpsnoise)
            self.graph.add(gtsam.GPSFactor(X(i), utm, gpsnoise))

    def add_prior_factor(self, T_prior_x_i, i, noise_type):
        """
        Estimating a prior factor on X(i), given the received estimation from other source.
        The main source here is the estimation of the prior with respect to the map, given
        the scanmatcher_map node.
        """
        noise = self.select_noise(noise_type)
        Tprior = gtsam.Pose3(T_prior_x_i.array)
        # add prior factor
        self.graph.push_back(gtsam.PriorFactorPose3(X(i), Tprior, noise))

    def optimize(self):
        print(50*'#')
        print('Optimize graphslam')
        self.isam.update(self.graph, self.initial_estimate)
        self.current_estimate = self.isam.calculateEstimate()
        self.initial_estimate.clear()
        # Reset Graph!!
        self.graph = gtsam.NonlinearFactorGraph()
        print('Optimize finished')
        print(50 * '#')

    def select_noise(self, noise_type):
        if noise_type == 'ODO':
            return self.ODO_NOISE
        elif noise_type == 'SM':
            return self.SM_NOISE
        elif noise_type == 'MAPSM':
            return self.MAPSM_NOISE
        elif noise_type == 'GPS':
            return self.GPS_NOISE

    def plot2D(self, plot_uncertainty_ellipse=False, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(0)
        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5,
                                          marginals.marginalCovariance(i))
            else:
                gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5, None)
            i += np.max([skip, 1])
        plt.pause(.01)

    def plot3D(self, plot_uncertainty_ellipse=False, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        fig = plt.figure(1)
        axes = fig.gca(projection='3d')
        plt.cla()
        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5,
                                                marginals.marginalCovariance(i))
            else:
                gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5, None)
            i += np.max([skip, 1])
        plt.pause(.01)

    def plot(self, plot3D=True, plot_uncertainty_ellipse=True, skip=1):
        """Print and plot incremental progress of the robot for 3D Pose SLAM using iSAM2."""
        # Compute the marginals for all states in the graph.
        if plot_uncertainty_ellipse:
            marginals = gtsam.Marginals(self.graph, self.current_estimate)

        # Plot the newly updated iSAM2 inference.
        if plot3D:
            fig = plt.figure(1)
            axes = fig.gca(projection='3d')
            plt.cla()
        else:
            fig = plt.figure(0)

        i = 0
        while self.current_estimate.exists(i):
            if plot_uncertainty_ellipse:
                if plot3D:
                    gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5,
                                                marginals.marginalCovariance(i))
                else:
                    gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5,
                                          marginals.marginalCovariance(i))
            else:
                if plot3D:
                    gtsam_plot.plot_pose3(0, self.current_estimate.atPose3(i), 0.5, None)
                else:
                    gtsam_plot.plot_pose2(0, self.current_estimate.atPose3(i), 0.5, None)

            i += np.max([skip, 1])
        plt.pause(.01)

    def plot_simple(self, plot3D=True, skip=1, gps_utm_readings=None):
        """
        Print and plot the result simply (no covariances or orientations)
        """
        # include estimates for poses X
        i = 0
        positions = []
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            positions.append(T.pos())
            i += np.max([skip, 1])
        positions = np.array(positions)
        landmarks = []
        for j in range(self.max_number_of_landmarks):
            if self.current_estimate.exists(L(j)):
                ce = self.current_estimate.atPose3(L(j))
                T = HomogeneousMatrix(ce.matrix())
                landmarks.append(T.pos())
        landmarks = np.array(landmarks)
        if plot3D:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(5)
            axes = fig.gca(projection='3d')
            plt.cla()
            if len(positions):
                axes.scatter(positions[:, 0], positions[:, 1], positions[:, 2], marker='.', color='blue')
            if len(landmarks) > 0:
                axes.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], marker='o', color='green')
            if gps_utm_readings is not None and len(gps_utm_readings) > 0:
                gps_utm_readings = np.array(gps_utm_readings)
                axes.scatter(gps_utm_readings[:, 0], gps_utm_readings[:, 1], gps_utm_readings[:, 2], marker='o', color='red')
            axes.legend()
        else:
            # Plot the newly updated iSAM2 inference.
            fig = plt.figure(0)
            plt.cla()
            if len(positions):
                plt.scatter(positions[:, 0], positions[:, 1], marker='.', color='blue')
            if len(landmarks) > 0:
                plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='o', color='green')
            if gps_utm_readings is not None and len(gps_utm_readings) > 0:
                gps_utm_readings = np.array(gps_utm_readings)
                plt.scatter(gps_utm_readings[:, 0], gps_utm_readings[:, 1], marker='o', color='red')
            plt.xlabel('X (m, UTM)')
            plt.ylabel('Y (m, UTM)')
        plt.pause(0.00001)

    def get_solution(self):
        return self.current_estimate

    def get_solution_index(self, index):
        if self.current_estimate.exists(X(index)):
            ce = self.current_estimate.atPose3(X(index))
            T = HomogeneousMatrix(ce.matrix())
        else:
            return None
        return T

    def get_solution_transforms(self):
        """
        This returns the states X as a homogeneous transform matrix.
        In this particular example, the state is represented as the position of the GPS on top of the robot.
        Using shorthand for X(i) (state at i)
        """
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T)
            i += 1
        return solution_transforms

    def get_solution_transforms_lidar(self):
        """
        This returns the states X as a homogeneous transform matrix.
        In this particular example, the state is represented as the position of the GPS on top of the robot.
        We transform this state to the center of the LiDAR.
        Using shorthand for X(i) (state at i)
        """
        solution_transforms = []
        i = 0
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            solution_transforms.append(T*self.Tlidar_gps.inv())
            i += 1
        return solution_transforms

    def get_solution_transforms_landmarks(self):
        """
        Using shorthand for L(j) (landmark j)
        """
        solution_transforms = []
        # landmarks ids
        landmark_ids = []
        for i in range(self.max_number_of_landmarks):
            if self.current_estimate.exists(L(i)):
                ce = self.current_estimate.atPose3(L(i))
                T = HomogeneousMatrix(ce.matrix())
                solution_transforms.append(T)
                # Using i as aruco_id identifier
                landmark_ids.append(i)
            i += 1
        return solution_transforms, landmark_ids

    def get_solution_positions(self):
        positions = []
        i = 0
        # fill a vector with all positions
        while self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            T = HomogeneousMatrix(ce.matrix())
            positions.append(T.pos())
            i += 1
        return np.array(positions)

    def get_relative_solution_transform(self, i, j):
        """
        Get the relative transformation between solutions i and j
        """
        if self.current_estimate.exists(X(i)):
            ce = self.current_estimate.atPose3(X(i))
            Ti = HomogeneousMatrix(ce.matrix())
        if self.current_estimate.exists(X(j)):
            ce = self.current_estimate.atPose3(X(i))
            Tj = HomogeneousMatrix(ce.matrix())
        Tij = Ti.inv() * Tj
        return Tij

    def save_solution(self, scan_times, directory):
        """
        Save the map.
        Saving a number of poses. Two reference systems:
        a) GPS reference system
        b) LiDAR reference system
        """
        euroc_read = EurocReader(directory=directory)
        global_transforms_gps = self.get_solution_transforms()
        global_transforms_lidar = self.get_solution_transforms_lidar()
        global_transforms_landmarks, landmark_ids = self.get_solution_transforms_landmarks()
        euroc_read.save_transforms_as_csv(scan_times, global_transforms_gps,
                                          filename='/robot0/SLAM/solution_graphslam_gps.csv')
        euroc_read.save_transforms_as_csv(scan_times, global_transforms_lidar,
                                          filename='/robot0/SLAM/solution_graphslam_lidar.csv')
        euroc_read.save_landmarks_as_csv(landmark_ids=landmark_ids, transforms=global_transforms_landmarks,
                                         filename='/robot0/SLAM/solution_graphslam_landmarks.csv')

    def plot_loop_closings(self, triplets):
        fig = plt.figure(3)
        positions = self.get_solution_positions()
        # Extract points
        xs, ys, zs = [], [], []
        for position in positions:
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])
        plt.scatter(xs, ys, c='b', marker='o')  # Plot poses

        # Plot edges
        for triplet in triplets:
            p1, p2, p3 = positions[triplet[0]], positions[triplet[1]], positions[triplet[2]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='g', linewidth=2)
            plt.plot([p1[0], p3[0]], [p1[1], p3[1]], color='k', linewidth=2)
            plt.plot([p2[0], p3[0]], [p2[1], p3[1]], color='k', linewidth=2)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("3D SLAM Graph with Loop Closures")
        plt.pause(0.1)

