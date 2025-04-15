import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from config import PARAMETERS
from graphSLAM.loopclosing import LoopClosing
from scanmatcher.scanmatcher import ScanMatcher
from gtsam.symbol_shorthand import X, L
import matplotlib.pyplot as plt


def compute_relative_transformation(lidarscanarray, posesarray, i, j, T0_gps):
    """
    Gets the times at the LiDAR observations at i and i+1.
    Gets the interpolated odometry values at those times.
    Computes an Homogeneous transform and computes the relative transformation Tij
    The T0gps transformation is considered. We are transforming from the LiDAR odometry to the GPS odometry (p. e.
    mounted to the front of the robot).
    This severs as a initial estimation for the ScanMatcher
    """
    delta_threshold_s = PARAMETERS.config.get('odo').get('delta_threshold_s')
    timei = lidarscanarray.get_time(i)
    timej = lidarscanarray.get_time(j)
    odoi = posesarray.interpolated_pose_at_time(timestamp=timei, delta_threshold_s=delta_threshold_s)
    odoj = posesarray.interpolated_pose_at_time(timestamp=timej)
    Ti = odoi.T()*T0_gps
    Tj = odoj.T()*T0_gps
    Tij = Ti.inv()*Tj
    return Tij


def process_odometry(graphslam, odoobsarray, smobsarray, lidarscanarray):
    """
    Add edge relations to the map
    """
    Tlidar_gps = graphslam.Tlidar_gps
    skip_optimization = graphslam.skip_optimization
    base_time = lidarscanarray.get_time(0)
    edges_odo = []
    #################################################################################################
    # loop through all edges first, include relative measurements such as odometry and scanmatching
    #################################################################################################
    for i in range(len(lidarscanarray) - 1):
        # i, i+1 edges.
        print('ADDING EDGE (i, j): (', i, ',', i + 1, ')')
        print('At experiment times i: ', (lidarscanarray.get_time(i) - base_time) / 1e9)
        print('At experiment times i+1: ', (lidarscanarray.get_time(i + 1) - base_time) / 1e9)
        atb_odo = compute_relative_transformation(lidarscanarray=lidarscanarray, posesarray=odoobsarray, i=i, j=i + 1,
                                                  T0_gps=Tlidar_gps)
        atb_sm = compute_relative_transformation(lidarscanarray=lidarscanarray, posesarray=smobsarray, i=i, j=i + 1,
                                                 T0_gps=Tlidar_gps)
        # create the initial estimate of node i+1 using SM
        graphslam.add_initial_estimate(atb_sm, i + 1)
        # graphslam.add_initial_estimate(atb_odo, i + 1)
        # add edge observations between vertices. We are adding a binary factor between a newly observed state and
        # the previous state. Using scanmatching odometry and raw odometry
        graphslam.add_edge(atb_sm, i, i + 1, 'SM')
        graphslam.add_edge(atb_odo, i, i + 1, 'ODO')
        edges_odo.append([i, i+1])
        # if i % skip_optimization == 0:
        #     print('GRAPHSLAM OPTIMIZE')
        #     print(50 * '*')
        #     graphslam.optimize()
        #     graphslam.plot_simple(plot3D=False)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    # graphslam.plot_simple(plot3D=True)
    return np.array(edges_odo)


def process_gps(graphslam, gpsobsarray, lidarscanarray):
    skip_optimization = graphslam.skip_optimization
    delta_threshold_s = PARAMETERS.config.get('gps').get('delta_threshold_s')
    utmfactors = []
    for i in range(len(lidarscanarray)):
        lidar_time = lidarscanarray.get_time(index=i)
        # given the lidar time, find the two closest GPS observations and get an interpolated GPS value
        gps_interp_reading = gpsobsarray.interpolated_utm_at_time(timestamp=lidar_time,
                                                                  delta_threshold_s=delta_threshold_s)
        if gps_interp_reading is not None:
            print('*** Adding GPS estimation at pose i: ', i)
            utmfactors.append([gps_interp_reading.x, gps_interp_reading.y, gps_interp_reading.altitude, i])
            graphslam.add_GPSfactor(utmx=gps_interp_reading.x, utmy=gps_interp_reading.y,
                                    utmaltitude=gps_interp_reading.altitude,
                                    gpsnoise=np.sqrt(gps_interp_reading.covariance),
                                    i=i)
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False, gps_utm_readings=utmfactors)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False, gps_utm_readings=utmfactors)
    # graphslam.plot_simple(plot3D=True, gps_utm_readings=utmfactors)
    return utmfactors


def process_aruco_landmarks(graphslam, arucoobsarray, lidarscanarray):
    # Filter ARUCO readings.
    Tgps_lidar = graphslam.Tgps_lidar
    Tlidar_cam = graphslam.Tlidar_cam
    skip_optimization = graphslam.skip_optimization
    delta_threshold_s = PARAMETERS.config.get('aruco').get('delta_threshold_s')
    # errors from ARUCO observations in the GPS reference system (X towards ARUCO).
    aruco_sigma_alpha = PARAMETERS.config.get('aruco').get('sigma_alpha')
    aruco_sigma_xy = PARAMETERS.config.get('aruco').get('sigma_xy')
    aruco_sigma_z = PARAMETERS.config.get('aruco').get('sigma_z')
    landmark_edges = []
    for j in range(len(arucoobsarray)):
        time_aruco = arucoobsarray.get_time(index=j)
        arucoobs = arucoobsarray.get(j)
        # the ARUCO observation from the camera reference system to the reference system placed on the GPS (the GPSfACTOR is directly applied)
        # transform the observation to the reference system on the GPS
        Tc_aruco = arucoobs.T()
        Tgps_aruco = Tgps_lidar*Tlidar_cam*Tc_aruco
        aruco_id = arucoobsarray.get_aruco_id(j)
        # The observation is attached to a pose X if the time is close to that correponding to the pose.
        # this is a simple solution, if the ARUCO observations are abundant it is highly possible to occur
        idx_lidar_graphslam, time_lidar_graphslam = lidarscanarray.get_index_closest_to_time(timestamp=time_aruco,
                                                                                             delta_threshold_s=delta_threshold_s)
        # if no time was found, simply continue the process
        if idx_lidar_graphslam is None:
            continue
        # if the landmark does not exist, create it from pose idx_lidar_graphslam
        # CAUTION: the landmarks is exactly numbered as the ARUCO identifier
        # if the landmarks does not exist --> then create it. We create the landmark estimate using the index of the
        # closest pose (idx_lidar_graphslam), the observation Tgps_aruco (expressed in the reference system of the GPS)
        # and the aruco_id itself
        if not graphslam.current_estimate.exists(L(aruco_id)):
            graphslam.add_initial_landmark_estimate(Tgps_aruco, idx_lidar_graphslam, aruco_id)
        else:
            # if the landmark exists, create edge between pose i and landmark aruco_id
            # sigmas stand for alpha, beta, gamma, sigmax, sigmay, sigmaz (sigmax is larger, since corresponds to Zc)
            graphslam.add_edge_pose_landmark(atb=Tgps_aruco, i=idx_lidar_graphslam, j=aruco_id,
                                             sigmas=np.array([aruco_sigma_alpha, aruco_sigma_alpha, aruco_sigma_alpha, aruco_sigma_z, aruco_sigma_xy, aruco_sigma_xy]))
            landmark_edges.append([aruco_id,  idx_lidar_graphslam])
        if j % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
            graphslam.plot_simple(plot3D=False)
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    # graphslam.plot_simple(plot3D=True)
    return np.array(landmark_edges)


def process_pairs_scanmatching(graphslam, lidarscanarray, pairs, n_pairs):
    result = []
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    Tlidar_gps = graphslam.Tlidar_gps
    k = 0
    # process randomly a number of pairs n_random
    n_random = n_pairs
    source_array = np.arange(len(pairs))
    random_elements = np.random.choice(source_array, n_random, replace=False)
    pairs = pairs[random_elements]
    for pair in pairs:
        print('Process pairs scanmatching: ', k, ' out of ', len(pairs))
        i = pair[0]
        j = pair[1]
        lidarscanarray.load_pointcloud(i)
        lidarscanarray.filter_points(i)
        lidarscanarray.estimate_normals(i)
        lidarscanarray.load_pointcloud(j)
        lidarscanarray.filter_points(j)
        lidarscanarray.estimate_normals(j)
        # current transforms
        Ti = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(i)).matrix())
        Tj = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(j)).matrix())
        # transform from GPS to Lidar
        Ti = Ti * Tlidar_gps.inv()
        Tj = Tj * Tlidar_gps.inv()
        Tij_0 = Ti.inv() * Tj
        Tij = scanmatcher.registration(i=i, j=j, Tij_0=Tij_0, show=False)
        lidarscanarray.unload_pointcloud(i)
        lidarscanarray.unload_pointcloud(j)
        result.append([i, j, Tij])
        k += 1
    return result


def process_triplets_scanmatching(graphslam, lidarscanarray, triplets):
    """
    Actually computing the transformation for each of the triplets with indexes (i, j, k)
    """
    result = []
    n = 0
    for triplet in triplets:
        print('Process pairs scanmatching: ', n, ' out of ', len(triplets))
        i = triplet[0]
        j = triplet[1]
        k = triplet[2]
        lidarscanarray.load_pointcloud(i)
        lidarscanarray.filter_points(i)
        lidarscanarray.estimate_normals(i)
        lidarscanarray.load_pointcloud(j)
        lidarscanarray.filter_points(j)
        lidarscanarray.estimate_normals(j)
        lidarscanarray.load_pointcloud(k)
        lidarscanarray.filter_points(k)
        lidarscanarray.estimate_normals(k)
        # the function compute_scanmatchin uses the initial estimation from the current estimation and then
        # performs scanmatching
        # CAUTION: the result is expressed in the LiDAR reference system, since it considers
        Tij = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=i, j=j)
        Tik = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=i, j=k)
        Tjk = compute_scanmathing(graphslam=graphslam, lidarscanarray=lidarscanarray, i=j, j=k)
        # remove lidar from memory
        lidarscanarray.unload_pointcloud(i)
        lidarscanarray.unload_pointcloud(j)
        lidarscanarray.unload_pointcloud(k)
        # result: the triplet and tranformations Tik, Tjk
        result.append([i, j, k, Tij, Tjk, Tik])
        n += 1
    return result


def compute_scanmathing(graphslam, lidarscanarray, i, j):
    scanmatcher = ScanMatcher(lidarscanarray=lidarscanarray)
    Tlidar_gps = graphslam.Tlidar_gps
    # current transforms. Compute initial transformation
    Ti = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(i)).matrix())
    Tj = HomogeneousMatrix(graphslam.current_estimate.atPose3(X(j)).matrix())
    # transform from GPS to Lidar
    Ti = Ti * Tlidar_gps.inv()
    Tj = Tj * Tlidar_gps.inv()
    # initial approximation from current state
    Tij_0 = Ti.inv() * Tj
    Tij = scanmatcher.registration(i=i, j=j, Tij_0=Tij_0, show=False)
    return Tij


def plot_sensors(odoarray, smarray, gpsarray):
    plt.figure()
    odo = odoarray.get_transforms()
    sm = smarray.get_transforms()
    gps = gpsarray.get_utm_positions()
    podo = []
    psm = []
    pgps = []
    for i in range(len(odo)):
        podo.append(odo[i].pos())
    for i in range(len(sm)):
        psm.append(sm[i].pos())
    for i in range(len(gps)):
        if gps[i].status >= 0:
            pgps.append([gps[i].x, gps[i].y, gps[i].altitude])
    podo = np.array(podo)
    psm = np.array(psm)
    pgps = np.array(pgps)
    plt.scatter(podo[:, 0], podo[:, 1], label='odo')
    plt.scatter(psm[:, 0], psm[:, 1], label='sm')
    plt.scatter(pgps[:, 0], pgps[:, 1], label='gps')
    plt.legend()
    plt.xlabel('X (m, UTM)')
    plt.ylabel('Y (m, UTM)')
    # plt.show()
    plt.pause(0.01)


def process_loop_closing_lidar(graphslam, lidarscanarray):
    """
        Find possible loop closings. The loop closing are computed as triplets (i, j, k) where i and j are close in the
        trajectory (with a low uncertainty) and k may be far. This allows to compute three transformations using ICP:
            - Tij (using as seed Tij0, from poses)
            - Tik (using a seed Tik0, from the estimated poses)
            - Tjk (using a seed Tjk0, from the estimated poses)
        It must be: Tij*Tik*Tjk.inv()=I. The three tran
        possesses a low uncertainty from the , whereas Tik and Tjk may have larger uncertainty. The observations
        The ARUCO observations lead to a trajectory that is mostly correct.
        In this particular case, the registration between the pointclouds using a vanilla ICP algorithm are fine.
        We plan to find candidates for loop closing in terms of triplets that can be filtered
    """
    # Find candidates for loop_closing. Computing triplets. Find unique triplets
    loop_closing = LoopClosing(graphslam=graphslam)
    # Compute unique triplets with scanmatching
    triplets = loop_closing.find_feasible_triplets()
    print('FOUND triplet candidates. A total of: ', len(triplets))
    print('COMPUTING CANDIDATES with scanmatching: ', len(triplets))
    graphslam.plot_loop_closings(triplets)
    # process_triplets_scanmathing: given the triplets, a scanmatchin procedure is performed to try to close the loops
    # this process highly depends on the ability of the ICP procedure to find a consistent registration (transformation)
    # the transformations are then filtered assuring that the triplets are consistent with Tij*Tjk*Tik.inv()==I
    triplets_transforms = process_triplets_scanmatching(graphslam=graphslam, lidarscanarray=lidarscanarray, triplets=triplets)
    print('Checking triplets. ', len(triplets_transforms))
    # Filter out wrong scanmatchings (the transformation may be wrong). Check Tij*Tjk*Tik.inv()==I
    triplets_transforms = loop_closing.check_triplet_transforms(triplet_transforms=triplets_transforms)
    print('After filtering!! Adding a total of triplets:', len(triplets_transforms))
    graphslam.plot_loop_closings(triplets_transforms)
    add_loopclosing_edges(graphslam, triplets_transforms)


def add_loopclosing_edges(graphslam, triplets_transforms):
    """
    Add edge relations to the map
    """
    Tlidar_gps = graphslam.Tlidar_gps
    Tgps_lidar = graphslam.Tgps_lidar
    skip_optimization = graphslam.skip_optimization
    #################################################################################################
    # loop through all edges and add them to the graph
    #################################################################################################
    n = 0
    for triplet in triplets_transforms:
        print('ADDING TRIPLET AS EDGES TO THE GRAPH. Adding triplet, ', n, 'out of ', len(triplets_transforms))
        # i, i+1 edges.
        i = triplet[0]
        j = triplet[1]
        k = triplet[2]
        Tij = triplet[3]
        Tjk = triplet[4]
        Tik = triplet[5]
        print('ADDING EDGE (i, j): (', i, ',', j, ')')
        print('ADDING EDGE (i, j): (', j, ',', k, ')')
        print('ADDING EDGE (i, j): (', i, ',', k, ')')
        # transfrom from the relative lidar reference system to the gps reference system
        # yes... this formula should be applied
        Tij = Tgps_lidar * Tij * Tlidar_gps
        Tjk = Tgps_lidar * Tjk * Tlidar_gps
        Tik = Tgps_lidar * Tik * Tlidar_gps
        # !! SMLC--> sm for loop closing may be noisier
        graphslam.add_edge(Tij, i, j, 'SM')
        graphslam.add_edge(Tjk, j, k, 'SM')
        graphslam.add_edge(Tik, i, k, 'SM')
        if i % skip_optimization == 0:
            print('GRAPHSLAM OPTIMIZE')
            print(50 * '*')
            graphslam.optimize()
        #     graphslam.plot_simple(plot3D=False)
        n += 1
    graphslam.optimize()
    graphslam.plot_simple(plot3D=False)
    # graphslam.plot_simple(plot3D=True)