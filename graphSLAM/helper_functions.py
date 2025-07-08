import numpy as np
# from artelib.homogeneousmatrix import HomogeneousMatrix
from config import PARAMETERS


def update_odo_observations(nodeloc, pose, timestamp):
    """
    ODO observations create a new state and a relation between two states.
    Each state is associated to a time that allows to create other relationships between
    the states: scanmatching, IMU... etc.
    """
    #################################################
    # integrate ODOMETRY measurements (interpolated)
    # find first odometry time
    #################################################
    if nodeloc.last_odom_pose is None:
        print("\033[91mCaution!!! No odometry received yet.\033[0m")
        return
    # check distance from the last odometry
    odo_ti = nodeloc.last_odom_pose
    odo_tj = pose
    # if the distance is larger or the angle is larger that... add pcd to buffer
    d_poses = PARAMETERS.config.get('graphslam').get('d_poses')
    th_poses = PARAMETERS.config.get('graphslam').get('th_poses')
    # Now, only add another pointcloud if the odometry is significantly moved
    d, th = compute_rel_distance(odo_ti, odo_tj)
    if d < d_poses and th < th_poses:
        return
    print('(d, th): ', d, th)
    print('Adding ODO STATE AND EDGES to the graph')
    print('Tiempo odometry', timestamp)
    Ti = nodeloc.last_odom_pose.T()
    Tj = pose.T()
    Tij = Ti.inv() * Tj
    print(50*'*')
    print('Print creating new state from ODO: (', nodeloc.current_key + 1, ')')
    print('Adding Graphslam INITIAL ODO edge: (', nodeloc.current_key, ',', nodeloc.current_key + 1, ')')
    print('Adding relative transformation: ')
    Tij.print_nice()
    print(50 * '*')
    print('Current key is: ', nodeloc.current_key)

    nodeloc.graphslam.add_initial_estimate(Tij, nodeloc.current_key + 1)
    nodeloc.graphslam.add_edge(Tij, nodeloc.current_key, nodeloc.current_key + 1, 'ODO')
    nodeloc.current_key += 1
    # append the index to the observation indices
    nodeloc.graphslam_observations_indices['ODO'].append(nodeloc.current_key)
    nodeloc.graphslam_times = np.append(nodeloc.graphslam_times, timestamp)
    # reset pose
    nodeloc.last_odom_pose = pose


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
    if nodeloc.current_key == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    ############################################
    # add sm observations as edges
    # the last edge "touched" is saved to avoid updating the same
    # edge indices constanly
    ############################################
    first_index_in_graphslam = nodeloc.last_processed_index['ODOSM']
    # given the last processed index in the graph. We iteratively follow
    # the rest of the nodes and try to include Scanmatching edge relations
    for i in range(first_index_in_graphslam, nodeloc.current_key - 1):
        time_graph1 = nodeloc.graphslam_times[i]
        time_graph2 = nodeloc.graphslam_times[i + 1]
        # getting the closest pose in scanmatching by interpolation
        smodoi, _, case_typei = nodeloc.odom_sm_buffer.interpolated_pose_at_time_new(timestamp=time_graph1)
        smodoj, _, case_typej = nodeloc.odom_sm_buffer.interpolated_pose_at_time_new(timestamp=time_graph2)
        if case_typei < 2 or case_typej < 2:
            continue
        if smodoi is None or smodoj is None:
            # print('NO SMODO FOR these graphslam nodes, SKIPPING')
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
        # append the index to the observation indices
        nodeloc.graphslam_observations_indices['ODOSM'].append(i)
        print('finished processing ODOSM')
        print(50 * '?')
    nodeloc.optimization_index += 1



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
        gpsi, _ = nodeloc.gps_buffer.interpolated_gps_at_time(time_graph1, delta_threshold_s=1.0)
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
        # store the touched index in the graph
        nodeloc.graphslam_observations_indices['GPS'].append(i)
        # store the last index. Next time, the graph is iterated from this
        nodeloc.last_processed_index['GPS'] = i + 1
    nodeloc.optimization_index += 1


def update_prior_map_observations(nodeloc):
    """
    This loops through the observations on the map.
    Each observation is a prior localization/refinement on the map.
    """
    if len(nodeloc.map_sm_prior_buffer) == 0:
        print("\033[91mCaution!!! No SM GLOBAL PRIOR in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    print('UPDATING with last GLOBAL SCANMATCHING MAPSM observations')

    # iterate from the last processed index in the graph, look for GPS and add them to the graph
    first_index_in_graphslam = nodeloc.last_processed_index['MAPSM']
    # running through the nodes of the graph (non visited yet). Looking for prior observations at that time
    # the prior observations are obtained by interpolation, from a buffer of prior estimations (with respect
    # to the global map)
    for i in range(first_index_in_graphslam, nodeloc.current_key):
        # get the corresponding time
        timestamp_i_in_graphslam = nodeloc.graphslam_times[i]
        # find the interpolation that corresponds to that particular time in the graph
        prior_i, _, case_type = nodeloc.map_sm_prior_buffer.interpolated_pose_at_time_new(timestamp_i_in_graphslam)
        if case_type < 2:
            continue
        if prior_i is None:
            # print('No estimation found')
            continue
        Trobot_prior = prior_i.T()
        # add_prior_factor, this is the localization according to the map scanmatching node.
        nodeloc.graphslam.add_prior_factor(Trobot_prior, i, 'MAPSM')
        nodeloc.graphslam_observations_indices['MAPSM'].add(i)
        # caution, allowing for some search back in time!
        nodeloc.last_processed_index['MAPSM'] = i
    nodeloc.optimization_index += 1

# def update_aruco_observations(nodeloc):
#     #################################################
#     # integrate ARUCO OBSERVATIONS
#     # ARUCO observations use the existing ARUCO landmarks map. Given a relative observation,
#     # we can retrieve and observe the robot position by a inverse transformation
#     #################################################
#     if len(nodeloc.aruco_observations_buffer) == 0:
#         print("\033[91mCaution!!! No aruco observations in buffer yet.\033[0m")
#         return
#     if len(nodeloc.graphslam_times) == 0:
#         print("\033[91mCaution!!! No graph yet.\033[0m")
#         return
#     delta_threshold_s = 0.05
#     timi_aruco_ini = nodeloc.aruco_observations_buffer.get_last_processed_time()
#     # look for the times in the graph that can be updated with new information
#     indices = np.where(nodeloc.graphslam_times > timi_aruco_ini)[0]
#     first_index = indices[0] if indices.size > 0 else None
#     if first_index is None:
#         print("\033[91mCaution!!! No times in graph with corresponding aruco time.\033[0m")
#         return
#     # running through the nodes of the graph (non visited yet)
#     # caution, the ARUCO observation is associated to any time that is found to be close
#     # to any time in the graph (close in time, i. e. 0.05 seconds, for example)
#     for i in range(first_index, len(nodeloc.graphslam_times)):
#         time_aruco_i = nodeloc.graphslam_times[i]
#         # get the relative transformation (stored as Pose) that is closest to that time
#         aruco_transform_i = nodeloc.aruco_observations_buffer.get_closest_pose_at_time(time_aruco_i,
#                                                                                        delta_threshold_s=delta_threshold_s)
#         index_aruco = nodeloc.aruco_observations_buffer.get_closest_index_to_time(time_aruco_i,
#                                                                                   delta_threshold_s=delta_threshold_s)
#         aruco_id = nodeloc.aruco_observations_ids[index_aruco]
#         # reset proc time
#         nodeloc.aruco_observations_buffer.last_processed_time = time_aruco_i + delta_threshold_s
#         if aruco_transform_i is None:
#             continue
#         # add a GPS factor on node i of the graph.aution
#         print('ADD ARUCO FACTOR!!!')
#         Tca = aruco_transform_i
#         Trobot = nodeloc.map.localize_with_aruco(Tca.T(), aruco_id, Tlidar_cam=nodeloc.graphslam.Tlidar_cam)
#         # This may happen if the ARUCO_ID is not found in the map
#         if Trobot is None:
#             continue
#         # add_prior_factor, aruco transform i, aruco_id
#         nodeloc.graphslam.add_prior_factor_aruco(Trobot, i)


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

