import numpy as np
from config import PARAMETERS


def update_sm_observations(nodeloc):
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


def update_odo_observations(nodeloc):
    #################################################
    # integrate ODOMETRY measurements (interpolated)
    # find first odometry time
    #################################################
    if len(nodeloc.odom_buffer) == 0:
        print("\033[91mCaution!!! No odometry in buffer yet.\033[0m")
        return
    if len(nodeloc.graphslam_times) == 0:
        print("\033[91mCaution!!! No graph yet.\033[0m")
        return
    first_index_in_graphslam = nodeloc.last_processed_index['ODO']
    print('Adding ODO EDGES to the graph')
    print(50*'?')
    # running through the nodes of the graph
    # for each node in the graph, look for corresponding times
    for i in range(first_index_in_graphslam, len(nodeloc.graphslam_times) - 1):
        time_graph1 = nodeloc.graphslam_times[i]
        time_graph2 = nodeloc.graphslam_times[i + 1]
        odoi = nodeloc.odom_buffer.interpolated_pose_at_time(time_graph1)
        odoj = nodeloc.odom_buffer.interpolated_pose_at_time(time_graph2)
        if odoi is None or odoj is None:
            print('NO ODO FOR these graphslam nodes, SKIPPING')
            continue
        print('Tiempo odometro', time_graph1 - nodeloc.start_time)
        Ti = odoi.T()
        Tj = odoj.T()
        Tij = Ti.inv() * Tj
        print(50*'*')
        print('Adding Graphslam ODO edge: (', i, ',', i+1, ')')
        print(50*'*')
        nodeloc.graphslam.add_edge(Tij, i, i + 1, 'ODO')
        # update the last processed index
        nodeloc.last_processed_index['ODO'] = i + 1
    print('finished processing ODO')
    print(50 * '?')

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

