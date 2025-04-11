"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
import timeit

import rospy
# import tf
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped
import gtsam
# from gtsam import Pose3, Rot3, Point3, Values, NonlinearFactorGraph, noiseModel
from gtsam import BetweenFactorPose3, PriorFactorPose3, PriorFactorPoint3, ISAM2, ISAM2Params
from gtsam.symbol_shorthand import X  # Pose keys
from geometry_msgs.msg import PoseStamped
from artelib.homogeneousmatrix import HomogeneousMatrix
from artelib.quaternion import Quaternion
from artelib.vector import Vector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tools.gpsconversions import gps2utm

from config import PARAMETERS

fig, ax = plt.subplots()
canvas = FigureCanvas(fig)

positions = []

class LocalizationNode:
    def __init__(self):
        print('Initializing localization node!')
        rospy.init_node('localization_node')
        print('Subscribing to odo, gnss')
        # Subscriptions
        rospy.Subscriber('/husky_velocity_controller/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/gnss/fix', NavSatFix, self.gps_callback)

        # Set up a timer to periodically update the plot
        rospy.Timer(rospy.Duration(1), self.timer_callback)

        # Publisher
        self.pub = rospy.Publisher('/localized_pose', Odometry, queue_size=10)

        # GTSAM setup
        self.isam = ISAM2(ISAM2Params())
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()

        self.prev_pose = None
        self.prev_time = None
        self.current_key = 0

        # Noise models
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05] * 6))  # x, y, z, roll, pitch, yaw
        self.gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, 3.0)  # GPS std dev ~1m

        # Latest GPS point
        self.latest_gps = None

        # self.positions = []
        self.skip_optimization = 500
        self.start_time = 0
        self.current_time = 0
        rospy.loginfo("ISAM2 Localization Node Initialized.")

        # plt.figure(0)
        # plt.ion()
        # fig = plt.figure()

    def odom_callback(self, msg):
        print('Received odo measurement')
        print('Adding current_key', self.current_key)
        timestamp = msg.header.stamp.to_sec()
        pose = self.convert_pose(msg.pose.pose)
        self.current_time = timestamp
        print('Current experiment time (s): ', (self.current_time-self.start_time))
        if self.current_key == 0:
            # First pose: add prior
            self.graph.add(PriorFactorPose3(X(0), pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))))
            self.initial_estimate.insert(X(0), pose)
            self.prev_pose = pose
            self.prev_time = timestamp
            self.update_isam()
            self.publish_pose(pose)
            self.current_key += 1
            self.start_time = timestamp
            return

        # Compute odometry delta
        delta = self.prev_pose.between(pose)
        self.graph.add(BetweenFactorPose3(X(self.current_key - 1), X(self.current_key), delta, self.odom_noise))
        self.initial_estimate.insert(X(self.current_key), pose)

        # Add GPS factor if available
        if self.latest_gps is not None:
            utm = [self.latest_gps['x'], self.latest_gps['y'],
                          self.latest_gps['altitude']]
            utm = gtsam.Point3(*utm)
            # self.graph.add(PriorFactorPoint3(gtsam.symbol(ord('X'), self.current_key), point, self.gps_noise))
            # self.graph.add(PriorFactorPoint3(gtsam.symbol(ord('X'), self.current_key), point, self.gps_noise))
            self.graph.add(gtsam.GPSFactor(X(self.current_key), utm, self.gps_noise))
            print(30*'**')
            print('ADDED UTM POSITION!!!')
            # reset
            self.latest_gps = None

        print('Updating iSAM')

        # if self.current_key % self.skip_optimization == 0:
        # Update iSAM
        self.update_isam()


        estimate = self.isam.calculateEstimate()
        pose_est = estimate.atPose3(X(self.current_key))
        self.publish_pose(pose_est)
        positions.append([pose_est.x(), pose_est.y()])

        # Prepare for next
        self.prev_pose = pose
        self.prev_time = timestamp
        self.current_key += 1


    def gps_callback(self, msg):
        # Convert lat/lon to dummy XYZ — replace with ENU or UTM in real use
        # self.latest_gps = [msg.latitude, msg.longitude, msg.altitude]

        self.latest_gps = convert_and_filter_gps(msg)
        if self.latest_gps is None:
            print("Received non-valid GPS reading!")
            return
        print("Received valid GPS reading!!")

        # self.latest_gps =

    def update_isam(self):
        start = timeit.timeit()
        try:
            self.isam.update(self.graph, self.initial_estimate)
            self.graph = gtsam.NonlinearFactorGraph()
            self.initial_estimate = gtsam.Values()
        except RuntimeError as e:
            rospy.logerr(f"iSAM2 update failed: {e}")
        end = timeit.timeit()
        print('Update ISAM took: ', end-start)

    def convert_pose(self, ros_pose):
        pos = ros_pose.position
        ori = ros_pose.orientation
        Q = Quaternion(qx=ori.x, qy=ori.y, qz=ori.z, qw=ori.w)
        T = HomogeneousMatrix(Vector([pos.x, pos.y, pos.z]), Q)
        return gtsam.Pose3(T.toarray())


    def publish_pose(self, pose):
        print('Publishing last pose:')
        # print(pose)
        msg = Odometry()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = pose.x()
        msg.pose.pose.position.y = pose.y()
        msg.pose.pose.position.z = pose.z()
        q = pose.rotation().toQuaternion()
        msg.pose.pose.orientation.x = q.x()
        msg.pose.pose.orientation.y = q.y()
        msg.pose.pose.orientation.z = q.z()
        msg.pose.pose.orientation.w = q.w()
        self.pub.publish(msg)

    def timer_callback(self, event):
        # plt.figure()
        if len(positions) == 0:
            return
        pos = np.array(positions)
        ax.clear()
        # x.plot(pos)
        ax.scatter(pos[:, 0], pos[:, 1], marker='.')
        canvas.print_figure('plot.png', bbox_inches='tight')

    def run(self):
        rospy.spin()

def convert_and_filter_gps(msg):
    max_sigma_xy = PARAMETERS.config.get('gps').get('max_sigma_xy')
    min_status = PARAMETERS.config.get('gps').get('min_status')
    if msg.status.status < min_status:
        return None
    sigma_xy = np.sqrt(msg.position_covariance[0])
    if sigma_xy > max_sigma_xy:
        return None
    df_gps = {}
    df_gps['latitude'] = msg.latitude
    df_gps['longitude'] = msg.longitude
    df_gps['altitude'] = msg.altitude
    # status = df_gps['status']
    # base reference system
    config_ref = {}
    config_ref['latitude'] = PARAMETERS.config.get('gps').get('utm_reference').get('latitude')
    config_ref['longitude'] = PARAMETERS.config.get('gps').get('utm_reference').get('longitude')
    config_ref['altitude'] = PARAMETERS.config.get('gps').get('utm_reference').get('altitude')
    df_utm = gps2utm(df_gps, config_ref)
    # convert to utm
    return df_utm


if __name__ == "__main__":
    filename = '/media/arvc/INTENSO/DATASETS/INDOOR_OUTDOOR/IO2-2025-03-25-16-54-17.bag'
    node = LocalizationNode()
    node.run()


