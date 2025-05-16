import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from config import PARAMETERS


def update_sm_observations(nodeloc):
    """
    # SM observations create a new state and a relation between two states.
    SM observations create relationships between states. The creation of new nodes in the graph is done
    with the odometry, which is faster.
    """
    print('UPDATING with last SM observations')
    if len(nodeloc.odom_sm_buffer) == 0:
        print("\033[91mCaution!!! No odometry SM in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    ############################################
    # add sm observations as edges
    # all sm observations are removed afterwards
    ############################################
    first_index_in_graphslam = nodeloc.last_processed_index['ODOSM']
    # given the last processed index in the graph. We iteratively follow
    # the rest of the nodes and try to include Scanmatching edge relations
    for i in range(first_index_in_graphslam, len(nodeloc.graphslam_times) - 1):
        time_graph1 = nodeloc.graphslam_times[i]
        time_graph2 = nodeloc.graphslam_times[i + 1]
        smodoi, _ = nodeloc.odom_sm_buffer.interpolated_pose_at_time(time_graph1)
        smodoj, _ = nodeloc.odom_sm_buffer.interpolated_pose_at_time(time_graph2)
        if smodoi is None or smodoj is None:
            print('NO SMODO FOR these graphslam nodes, SKIPPING')
            continue
        print('Tiempo odometro scanmatcher', time_graph1 - nodeloc.start_time)
        Ti = smodoi.T()
        Tj = smodoj.T()
        Tij = Ti.inv() * Tj
        print(50*'*')
        print('Adding Graphslam ODOSM edge: (', i, ',', i+1, ')')
        print(50*'*')
        nodeloc.graphslam.add_edge(Tij, i, i + 1, 'ODOSM')
        # update the last processed index
        nodeloc.last_processed_index['ODOSM'] = i + 1
        print('finished processing ODOSM')
        print(50 * '?')
    # for i in range(len(nodeloc.odom_sm_buffer) - 1):
    #     print('Tiempo scanmatcher', nodeloc.odom_sm_buffer.times[i] - nodeloc.start_time)
    #     Ti = nodeloc.odom_sm_buffer[i].T()
    #     Tj = nodeloc.odom_sm_buffer[i + 1].T()
    #     Tij = Ti.inv() * Tj
    #     print(50*'*')
    #     print('Print creating new state: (', nodeloc.current_key + 1, ')')
    #     print('Adding Graphslam INITIAL SMODO edge: (', nodeloc.current_key, ',', nodeloc.current_key + 1, ')')
    #     print(50 * '*')
    #     nodeloc.graphslam.add_initial_estimate(Tij, nodeloc.current_key + 1)
    #     nodeloc.graphslam.add_edge(Tij, nodeloc.current_key, nodeloc.current_key + 1, 'SMODO')
    #     nodeloc.current_key += 1
    #     next_time = nodeloc.odom_sm_buffer.times[i + 1]
    #     nodeloc.graphslam_times = np.append(nodeloc.graphslam_times, next_time)
    #     k += 1
    # # removed used odom_sm observations
    # for j in range(k):
    #     nodeloc.odom_sm_buffer.popleft()


def update_odo_observations(nodeloc, pose, timestamp):
    """
    ODO observations create a new state and a relation between two states.
    """
    #################################################
    # integrate ODOMETRY measurements (interpolated)
    # find first odometry time
    #################################################
    # if len(nodeloc.odom_buffer) == 0:
    #     print("\033[91mCaution!!! No odometry in buffer yet.\033[0m")
    #     return
    if nodeloc.last_odom_pose is None:
        print("\033[91mCaution!!! No odometry received yet.\033[0m")
        return
    # first_index_in_graphslam = nodeloc.last_processed_index['ODO']
    odo_ti = nodeloc.last_odom_pose
    odo_tj = pose
    # if the distance is larger or the angle is larger that... add pcd to buffer
    # d_poses = PARAMETERS.config.get('scanmatcher').get('d_poses')
    # th_poses = PARAMETERS.config.get('scanmatcher').get('th_poses')
    # Now, only add another pointcloud if the odometry is significantly moved
    d, th = compute_rel_distance(odo_ti, odo_tj)
    if d < 0.3 and th < 0.1:
        # print('Not enough distance traveled. ')
        # print('No new nodes created in graph.')
        return
    print('Adding ODO STATE AND EDGES to the graph')
    print(50*'?')
    # for i in range(len(nodeloc.odo_buffer) - 1):
    print('Tiempo odometry', timestamp)
    Ti = nodeloc.last_odom_pose.T()
    Tj = pose.T()
    Tij = Ti.inv() * Tj
    print(50*'*')
    print('Print creating new state from ODO: (', nodeloc.current_key + 1, ')')
    print('Adding Graphslam INITIAL ODO edge: (', nodeloc.current_key, ',', nodeloc.current_key + 1, ')')
    print(50 * '*')
    nodeloc.graphslam.add_initial_estimate(Tij, nodeloc.current_key + 1)
    nodeloc.graphslam.add_edge(Tij, nodeloc.current_key, nodeloc.current_key + 1, 'ODO')
    nodeloc.current_key += 1
    # next_time = nodeloc.odom_sm_buffer.times[i + 1]
    nodeloc.graphslam_times = np.append(nodeloc.graphslam_times, timestamp)
    # reset pose
    nodeloc.last_odom_pose = pose
        # k += 1
    # # running through the nodes of the graph
    # # for each node in the graph, look for corresponding times
    # for i in range(first_index_in_graphslam, len(nodeloc.graphslam_times) - 1):
    #     time_graph1 = nodeloc.graphslam_times[i]
    #     time_graph2 = nodeloc.graphslam_times[i + 1]
    #     odoi, _ = nodeloc.odom_buffer.interpolated_pose_at_time(time_graph1)
    #     odoj, _ = nodeloc.odom_buffer.interpolated_pose_at_time(time_graph2)
    #     if odoi is None or odoj is None:
    #         print('NO ODO FOR these graphslam nodes, SKIPPING')
    #         continue
    #     print('Tiempo odometro', time_graph1 - nodeloc.start_time)
    #     Ti = odoi.T()
    #     Tj = odoj.T()
    #     Tij = Ti.inv() * Tj
    #     print(50*'*')
    #     print('Adding Graphslam ODO edge: (', i, ',', i+1, ')')
    #     print(50*'*')
    #     nodeloc.graphslam.add_edge(Tij, i, i + 1, 'ODO')
    #     # update the last processed index
    #     nodeloc.last_processed_index['ODO'] = i + 1
    # print('finished processing ODO')
    # print(50 * '?')

def update_gps_observations(nodeloc):
    #################################################
    # integrate GPS measurements (interpolated)
    #################################################
    if len(nodeloc.gps_buffer) == 0:
        print("\033[91mCaution!!! No gps in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    # iterate from the last processed index in the graph, look for GPS and add them to the graph
    first_index_in_graphslam = nodeloc.last_processed_index['GPS']
    # running through the nodes of the graph (non visited yet). Looking for gps observations at that time
    for i in range(first_index_in_graphslam, len(nodeloc.graphslam_times)):
        time_graph1 = nodeloc.graphslam_times[i]
        gpsi = nodeloc.gps_buffer.interpolated_gps_at_time(time_graph1, delta_threshold_s=1.0)
        # reset proc time
        nodeloc.gps_buffer.last_processed_time = time_graph1
        if gpsi is None:
            print('NO GPS FOR this graphslam node, SKIPPING')
            continue
        print('Tiempo GPS', time_graph1 - nodeloc.start_time)
        # add a GPS factor on node i of the graph.aution
        print('ADD GPS FACTOR!!!')
        nodeloc.graphslam.add_GPSfactor(utmx=gpsi.x,
                                        utmy=gpsi.y,
                                        utmaltitude=gpsi.altitude,
                                        gpsnoise=np.sqrt(gpsi.position_covariance),
                                        i=i)
        nodeloc.last_processed_index['GPS'] = i + 1


def update_global_sm_observations(nodeloc):
    if len(nodeloc.odo_sm_buffer) == 0:
        print("\033[91mCaution!!! No SM ODO in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return

    first_index_in_graphslam = nodeloc.last_processed_index['GLOBALSM']
    # running through the nodes of the graph (non visited yet). Looking for relative scanmatching to the map
    for i in range(first_index_in_graphslam, len(nodeloc.graphslam_times)):
        # Get the solution i on the graph
        T0i = nodeloc.graphslam.get_solution_index(i)
        time_i = nodeloc.graphslam_times[i]
        # retrieve the closest pointcloud in the buffer associated to time_i
        current_pcd = nodeloc.pcdbuffer.get_closest_to_time(timestamp=time_i, delta_threshold_s=1.0)

        # get the closest pose on the map.
        map_posej, time_map = nodeloc.map.get_closest_pose(timestamp=T0i.pos(), delta_threshold_s=1.0)
        if map_posej is None:
            return


        # compute the initial relative transformation.


        # compute registration


        # compute global transformation T0i= T0j*Tij.inv()

        # add a prior factor to T0i

    # current_pcd_time = nodeloc.pcdbuffer.times[i]
    #
    # current_pcd_time = current_pcd_time + 0.5

    # current_time = rospy.Time.now().to_sec()
    # diff = current_time-current_pcd_time
    # current test approach: find a map pose closest in time
    # needed approach: get pointclouds in the map closest in euclidean distance

    #     # continue
    # # get the closest pcd in the map
    # map_pcd, pointcloud_time = nodeloc.map.get_pcd_closest_to_time(timestamp=current_pcd_time,
    #                                                             delta_threshold_s=1.0)
    # diff = current_pcd_time - pointcloud_time
    # print('Diff in time: current pcd and map pcd: ', diff)
    #
    # # process the current pcd
    # current_pcd.down_sample(voxel_size=None)
    # current_pcd.filter_points()
    # current_pcd.estimate_normals()
    # # current_pcd.draw_cloud()
    #
    # # current the map pcd
    # map_pcd.load_pointcloud()
    # map_pcd.down_sample(voxel_size=None)
    # map_pcd.filter_points()
    # map_pcd.estimate_normals()
    # # map_pcd.draw_cloud()
    # # now, in this, test, consider that the initial transformation is the identity
    # # In the final approach: consider that the relative initial transformation is known
    # # Tij0 = HomogeneousMatrix(Vector([0.1, 0.1, 0]), Euler([0.1, 0.1, 0.1]))
    # Tij0 = HomogeneousMatrix()
    # Tij = nodeloc.scanmatcher.registration(current_pcd, map_pcd, Tij_0=Tij0, show=False)
    #
    # map_pcd.unload_pointcloud()
    # # the map pose (pointcloud)
    # T0j = map_pose.T()
    # # estimate the initial i
    # T0i = T0j * Tij.inv()
    # nodeloc.prior_estimations.append(T0i)
    # nodeloc.processed_map_poses.append(map_pose.T())


def update_aruco_observations(nodeloc):
    #################################################
    # integrate ARUCO OBSERVATIONS
    # ARUCO observations use the existing ARUCO landmarks map. Given a relative observation,
    # we can retrieve and observe the robot position by a inverse transformation
    #################################################
    if len(nodeloc.aruco_observations_buffer) == 0:
        print("\033[91mCaution!!! No aruco observations in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    delta_threshold_s = 0.05
    timi_aruco_ini = nodeloc.aruco_observations_buffer.get_last_processed_time()
    # look for the times in the graph that can be updated with new information
    indices = np.where(nodeloc.graphslam_times > timi_aruco_ini)[0]
    first_index = indices[0] if indices.size > 0 else None
    if first_index is None:
        print("\033[91mCaution!!! No times in graph with corresponding aruco time.\033[0m")
        return
    # running through the nodes of the graph (non visited yet)
    # caution, the ARUCO observation is associated to any time that is found to be close
    # to any time in the graph (close in time, i. e. 0.05 seconds, for example)
    for i in range(first_index, len(nodeloc.graphslam_times)):
        time_aruco_i = nodeloc.graphslam_times[i]
        # get the relative transformation (stored as Pose) that is closest to that time
        aruco_transform_i = nodeloc.aruco_observations_buffer.get_closest_pose_at_time(time_aruco_i,
                                                                                       delta_threshold_s=delta_threshold_s)
        index_aruco = nodeloc.aruco_observations_buffer.get_closest_index_to_time(time_aruco_i,
                                                                                  delta_threshold_s=delta_threshold_s)
        aruco_id = nodeloc.aruco_observations_ids[index_aruco]
        # reset proc time
        nodeloc.aruco_observations_buffer.last_processed_time = time_aruco_i + delta_threshold_s
        if aruco_transform_i is None:
            continue
        # add a GPS factor on node i of the graph.aution
        print('ADD ARUCO FACTOR!!!')
        Tca = aruco_transform_i
        Trobot = nodeloc.map.localize_with_aruco(Tca.T(), aruco_id, Tlidar_cam=nodeloc.graphslam.Tlidar_cam)
        # This may happen if the ARUCO_ID is not found in the map
        if Trobot is None:
            continue
        # add_prior_factor, aruco transform i, aruco_id
        nodeloc.graphslam.add_prior_factor_aruco(Trobot, i)


def compute_scanmatching_to_map(nodeloc):
    """
    SM observations create a new state and a relation between two states.
    """
    print('UPDATING with last SM observations')
    k = 0
    ############################################
    # add sm observations as edges
    # all sm observations are removed afterwards
    ############################################
    for i in range(len(nodeloc.odom_sm_buffer) - 1):
        print('Tiempo scanmatcher', nodeloc.odom_sm_buffer.times[i] - nodeloc.start_time)
        Ti = nodeloc.odom_sm_buffer[i].T()
        Tj = nodeloc.odom_sm_buffer[i + 1].T()
        Tij = Ti.inv() * Tj
        print(50*'*')
        print('Print creating new state: (', nodeloc.current_key + 1, ')')
        print('Adding Graphslam INITIAL SMODO edge: (', nodeloc.current_key, ',', nodeloc.current_key + 1, ')')
        print(50 * '*')
        nodeloc.graphslam.add_initial_estimate(Tij, nodeloc.current_key + 1)
        nodeloc.graphslam.add_edge(Tij, nodeloc.current_key, nodeloc.current_key + 1, 'SMODO')
        nodeloc.current_key += 1
        next_time = nodeloc.odom_sm_buffer.times[i + 1]
        nodeloc.graphslam_times = np.append(nodeloc.graphslam_times, next_time)
        k += 1
    # removed used odom_sm observations
    for j in range(k):
        nodeloc.odom_sm_buffer.popleft()


def compute_rel_distance(odo1, odo2):
    Ti = odo1.T()
    Tj = odo2.T()
    Tij = Ti.inv() * Tj
    d = np.linalg.norm(Tij.pos())
    e1 = np.linalg.norm(Tij.euler()[0].abg)
    e2 = np.linalg.norm(Tij.euler()[1].abg)
    theta = min(e1, e2)
    # print('Relative (d, theta):', d, theta)
    return d, theta


def filter_and_convert_gps_observations(gpsposition):
    """
    filter gps and convert to UTM
    """
    # filter gps readings
    if gpsposition.status < PARAMETERS.config.get('gps').get('min_status'):
        return None
    # filter gps readings with sigma greater than the max allowed for the problem.
    if np.sqrt(gpsposition.position_covariance[0]) > PARAMETERS.config.get('gps').get('max_sigma_xy'):
        return None
    # convert to UTM
    utm_ref = PARAMETERS.config.get('gps').get('utm_reference')
    utmposition = gpsposition.to_utm(utm_ref)
    return utmposition

