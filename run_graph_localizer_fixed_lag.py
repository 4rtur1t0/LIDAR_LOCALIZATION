"""
Using GTSAM in a GraphSLAM context.
We are integrating odometry, scanmatching odometry and (if present) GPS.
    The state X is the position and orientation frame of the robot, placed on the GPS sensor.

"""
import rospy
import tf
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseStamped
from gtsam import *
from gtsam.symbol_shorthand import X
from collections import deque
from datetime import timedelta

class FixedLagLocalizationNode:
    def __init__(self):
        rospy.init_node("fixed_lag_localization")

        # Publishers and subscribers
        rospy.Subscriber("/odom", Odometry, self.odom_callback)
        rospy.Subscriber("/gps/fix", NavSatFix, self.gps_callback)
        self.pose_pub = rospy.Publisher("/localized_pose", Odometry, queue_size=10)

        # Sliding window smoother
        self.lag = 2.0  # seconds
        self.smoother = FixedLagSmootherBatch(timedelta(seconds=self.lag), LevenbergMarquardtParams())

        # Buffers
        self.prev_pose = None
        self.current_key = 0
        self.prev_time = None

        # Noise models
        self.odom_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
        self.gps_noise = noiseModel.Isotropic.Sigma(3, 2.0)

        # Factor graph for each step
        self.graph = NonlinearFactorGraph()
        self.initial_estimates = Values()

    def odom_callback(self, msg):
        curr_time = rospy.Time.now().to_sec()

        # Convert ROS odometry to GTSAM Pose3
        pose = self.convert_pose(msg.pose.pose)

        if self.current_key == 0:
            # Add prior
            prior_noise = noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
            self.graph.add(PriorFactorPose3(X(0), pose, prior_noise))
            self.initial_estimates.insert(X(0), pose)
            self.prev_pose = pose
            self.prev_time = curr_time
            self.update_smoother(curr_time)
            self.current_key += 1
            return

        delta = self.prev_pose.between(pose)
        self.graph.add(BetweenFactorPose3(X(self.current_key - 1), X(self.current_key), delta, self.odom_noise))
        self.initial_estimates.insert(X(self.current_key), pose)

        self.prev_pose = pose
        self.prev_time = curr_time
        self.update_smoother(curr_time)
        self.current_key += 1

    def gps_callback(self, msg):
        if self.current_key == 0:
            return

        # Convert lat/lon/alt to a dummy XYZ (use ENU/UTM for real systems)
        point = Point3(msg.latitude, msg.longitude, msg.altitude)
        pose_key = X(self.current_key - 1)

        # GPS as PriorFactor on translation only
        gps_factor = PriorFactorPoint3(Pose3.translationKey(pose_key), point, self.gps_noise)
        self.graph.add(gps_factor)

    def update_smoother(self, timestamp):
        key_timestamp = {X(self.current_key): timestamp}

        self.smoother.update(self.graph, self.initial_estimates, key_timestamp)
        result = self.smoother.calculateEstimate()

        # Clear graph and estimates for next step
        self.graph = NonlinearFactorGraph()
        self.initial_estimates = Values()

        # Publish the latest pose
        if result.exists(X(self.current_key)):
            pose = result.atPose3(X(self.current_key))
            self.publish_pose(pose)

    def convert_pose(self, ros_pose):
        pos = ros_pose.position
        ori = ros_pose.orientation
        r = tf.transformations.quaternion_matrix([ori.x, ori.y, ori.z, ori.w])
        t = np.array([pos.x, pos.y, pos.z])
        T = np.eye(4)
        T[:3, :3] = r[:3, :3]
        T[:3, 3] = t
        return Pose3(T)

    def publish_pose(self, pose):
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
        self.pose_pub.publish(msg)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = FixedLagLocalizationNode()
    node.run()
