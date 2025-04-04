"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
import numpy as np
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.vector import Vector
from artelib.euler import Euler
from graphslam.loopclosing import LoopClosing
from graphslam.helper_functions import process_odometry, process_gps, process_aruco_landmarks, \
    process_triplets_scanmatching, plot_sensors, process_loop_closing_lidar
from lidarscanarray.lidarscanarray import LiDARScanArray
from observations.gpsarray import GPSArray
from observations.posesarray import PosesArray, ArucoPosesArray
import getopt
import sys
from graphslam.graphSLAM import GraphSLAM
import matplotlib.pyplot as plt
from map.map import Map
from session.session import Session


def find_options():
    argv = sys.argv[1:]
    euroc_path = None
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('python run_graphSLAM.py -i <euroc_directory>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python run_graphSLAM.py -i <euroc_directory>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            euroc_path = arg
    print('Input find_options directory is: ', euroc_path)
    return euroc_path


def load_experiment(directory):
    # odometry
    odoobsarray = PosesArray()
    odoobsarray.read_data(directory=directory, filename='/robot0/odom/data.csv')
    # scanmatcher
    smobsarray = PosesArray()
    smobsarray.read_data(directory=directory, filename='/robot0/scanmatcher/data.csv')
    # ARUCO observations. In the camera reference frame
    arucoobsarray = ArucoPosesArray()
    arucoobsarray.read_data(directory=directory, filename='/robot0/aruco/data.csv')
    # remove spurious ARUCO IDs
    arucoobsarray.filter_aruco_ids()
    # gpsobservations
    gpsobsarray = GPSArray()
    gpsobsarray.read_data(directory=directory, filename='/robot0/gps0/data.csv')
    gpsobsarray.read_config_ref(directory=directory)
    gpsobsarray.filter_measurements()
    # gpsobsarray.plot_xyz_utm()
    # Plot initial sensors as raw data
    # plot_sensors(odoarray=odoobsarray, smarray=smobsarray, gpsarray=gpsobsarray)
    # create scan Array, We are actually estimating the poses at which
    lidarscanarray = LiDARScanArray(directory=directory)
    lidarscanarray.read_parameters()
    lidarscanarray.read_data()
    # remove scans without corresponding odometry (in consequence, without scanmatching)
    lidarscanarray.remove_orphan_lidars(pose_array=odoobsarray)
    lidarscanarray.remove_orphan_lidars(pose_array=smobsarray)
    # load the scans according to the times, do not load the corresponding pointclouds
    lidarscanarray.add_lidar_scans()
    return odoobsarray, smobsarray, arucoobsarray, gpsobsarray, lidarscanarray


def initial_aruco_localization(session, map, **kwargs):
    # Tlidar_gps = kwargs.get('Tlidar_gps')
    # Tlidar_gps = kwargs.get('Tlidar_gps')
    T0 = None
    while True:
        observations = session.get_next_observations()
        for observation in observations:
            if observation[0] == 'ARUCO':
                # T0: Define the initial transformation (Prior for GraphSLAM)
                # T0 = HomogeneousMatrix()
                Tca = observation[1].T()
                aruco_id = observation[2]
                # localize with aruco, compute T0 for GraphLoc
                T0 = map.localize_with_aruco(Tca, aruco_id, **kwargs)
                break
        if T0 is None:
            print('ERROR: no ARUCO found for initial localization. Follow up!')
        else:
            return T0


def run_localizer():
    """
    The localizer loop
    """
    map_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'
    session_directory = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17'

    # Caution: Actually, we are estimating the position and orientation of the GPS at this position at the robot.
    # T LiDAR-GPS
    Tlidar_gps = HomogeneousMatrix(Vector([0.36, 0, -0.4]), Euler([0, 0, 0]))
    # T LiDAR-camera
    Tlidar_cam = HomogeneousMatrix(Vector([0, 0.17, 0]), Euler([0, np.pi / 2, -np.pi / 2]))
    # Try to visualize an ARUCO and compute T0. Define a low uncertainty prior


    # The map object.
    # Stores the pointclouds along with the poses. Each poincloud is identified by the time
    # Stores the ARUCO landmarks
    # may store more than one map. i.e. more than one path with the pointclouds

    # Functions.
    # Given a pose t, allows to find a set of pointclouds and compute transformations in triangle
    # alternatively, transformations between i and j
    # maybe threaded
    # --> copy the LoopClosing object
    map = Map()
    map.read_data(directory=map_directory)
    # methods
    # compute_transforms: given the state x and pointcloud pc, find 2-3 closest pointclouds
    # localize_with_aruco: given an aruco observation, compute an approximate location in the map.

    # load the observations
    odoobsarray, smobsarray, arucoobsarray, gpsobsarray, lidarscanarray = load_experiment(session_directory)
    # create an Experiment object.
    # stores all the times in a vector of times. The observations can be obtained at all the times or at different
    # timesteps. i.e. obtain odo, imu, LiDAR.
    session = Session(odo=odoobsarray, smodo=smobsarray, aruco=arucoobsarray, gps=gpsobsarray, lidar=lidarscanarray)
    session.init()

    T0 = initial_aruco_localization(session=session, map=map, Tlidar_gps=Tlidar_gps, Tlidar_cam=Tlidar_cam)


    # create the graphLocalizer
    # Add the landmarks as constant factors
    # given a odo measurement, add edge between t1 and t2
    #           Adds state X at time t2
    # given a pointcloud at t3, add edge between L(j) and X(t3)
    # given a GPS measurement at t3.
    #       add GPS factor
    # must keep an array of times

    # First: add data in a loop sequentially.
    # second: subscribe
    # create a function/object that returns all the times in the sensors in orders.
    # next, the object returns for each time
    graphloc = GraphLoc(T0=T0, Tlidar_gps=Tlidar_gps, Tlidar_cam=Tlidar_cam, skip_optimization=skip_optimization)
    graphloc.init_graph()
    graphloc.init_pointcloud_landmarks(map=map)
    # odo_times = odoobsarray.get_times()
    # for i in range(len(odo_times)):
    while True:
        observations = session.get_next_observations()
        if len(observations) == 0:
            break

    graphloc.save_solution(directory=directory, scan_times=lidarscanarray.get_times())


if __name__ == "__main__":
    directory = find_options()
    run_localizer()
