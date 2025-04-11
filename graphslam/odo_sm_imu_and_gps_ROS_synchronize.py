#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
import gtsam
from gtsam import symbol
from gtsam.utils import plot
from pyproj import Proj
import message_filters

class GTSAMLocalizer:
    def __init__(self):
        self.g = 9.81
        self.latest_time = None

        self.pose_key = lambda i: symbol('x', i)
        self.vel_key = lambda i: symbol('v', i)
        self.bias_key = lambda i: symbol('b', i)

        self.proj = Proj(proj='utm', zone=33, ellps='WGS84')  # Adjust zone!

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.isam = gtsam.ISAM2()

        self.imu_params = gtsam.PreintegrationParams.MakeSharedU(self.g)
        self.imu_params.setAccelerometerCovariance(np.eye(3)*0.1)
        self.imu_params.setGyroscopeCovariance(np.eye(3)*0.1)
        self.imu_params.setIntegrationCovariance(np.eye(3)*0.01)

        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1]*6))
        self.vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.gps_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)

        self.bias = gtsam.imuBias.ConstantBias()

        self.prev_state = gtsam.NavState(gtsam.Pose3(), np.zeros(3))
        self.imu_preint = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)

        self.current_index = 0
        self.start_time = rospy.Time.now().to_sec()

        self.graph.add(gtsam.PriorFactorPose3(self.pose_key(0), self.prev_state.pose(), self.pose_noise))
        self.graph.add(gtsam.PriorFactorVector(self.vel_key(0), self.prev_state.velocity(), self.vel_noise))
        self.graph.add(gtsam.PriorFactorConstantBias(self.bias_key(0), self.bias, self.bias_noise))

        self.initial_values.insert(self.pose_key(0), self.prev_state.pose())
        self.initial_values.insert(self.vel_key(0), self.prev_state.velocity())
        self.initial_values.insert(self.bias_key(0), self.bias)

        self.isam.update(self.graph, self.initial_values)

        # ROS Subscribers with synchronization
        imu_sub = message_filters.Subscriber("/imu/data", Imu)
        gps_sub = message_filters.Subscriber("/gps/fix", NavSatFix)
        wheel_sub = message_filters.Subscriber("/odom/wheel", Odometry)
        lidar_sub = message_filters.Subscriber("/lidar_odometry", Odometry)

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [imu_sub, gps_sub, wheel_sub, lidar_sub], queue_size=50, slop=0.05)
        self.ts.registerCallback(self.synchronized_callback)

        self.pose_pub = rospy.Publisher("/fused_pose", PoseStamped, queue_size=10)

    def assign_pose_index(self, timestamp):
        return int((timestamp.to_sec() - self.start_time) * 10)  # Assuming ~10Hz update rate

    def synchronized_callback(self, imu_msg, gps_msg, wheel_msg, lidar_msg):
        self.process_imu(imu_msg)
        self.process_gps(gps_msg)
        self.process_wheel(wheel_msg)
        self.process_lidar(lidar_msg)

    def process_imu(self, msg):
        acc = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        if self.latest_time:
            dt = (msg.header.stamp - self.latest_time).to_sec()
            self.imu_preint.integrateMeasurement(acc, gyro, dt)
        self.latest_time = msg.header.stamp

    def process_gps(self, msg):
        if not msg.status.status >= 0:
            return
        e, n = self.proj(msg.longitude, msg.latitude)
        pos = np.array([e, n, msg.altitude])

        i = self.current_index + 1
        predicted_state = self.imu_preint.predict(self.prev_state, self.bias)

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        graph.add(gtsam.ImuFactor(
            self.pose_key(self.current_index), self.vel_key(self.current_index),
            self.pose_key(i), self.vel_key(i),
            self.bias_key(self.current_index), self.imu_preint))

        graph.add(gtsam.BetweenFactorConstantBias(
            self.bias_key(self.current_index), self.bias_key(i),
            gtsam.imuBias.ConstantBias(), self.bias_noise))

        graph.add(gtsam.GPSFactor(self.pose_key(i), pos, self.gps_noise))

        values.insert(self.pose_key(i), predicted_state.pose())
        values.insert(self.vel_key(i), predicted_state.velocity())
        values.insert(self.bias_key(i), self.bias)

        self.isam.update(graph, values)
        result = self.isam.calculateEstimate()
        self.prev_state = gtsam.NavState(
            result.atPose3(self.pose_key(i)),
            result.atVector(self.vel_key(i)))
        self.bias = result.atConstantBias(self.bias_key(i))
        self.imu_preint.resetIntegrationAndSetBias(self.bias)
        self.current_index = i

        self.publish_pose(result.atPose3(self.pose_key(i)), msg.header.stamp)

    def process_wheel(self, msg):
        delta_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z),
            np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z]))
        self.graph.add(gtsam.BetweenFactorPose3(
            self.pose_key(self.current_index - 1), self.pose_key(self.current_index),
            delta_pose, self.pose_noise))

    def process_lidar(self, msg):
        delta_pose = gtsam.Pose3(
            gtsam.Rot3.Quaternion(
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z),
            np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z]))
        self.graph.add(gtsam.BetweenFactorPose3(
            self.pose_key(self.current_index - 1), self.pose_key(self.current_index),
            delta_pose, self.pose_noise))

    def publish_pose(self, pose, stamp):
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = pose.x()
        msg.pose.position.y = pose.y()
        msg.pose.position.z = pose.z()
        quat = pose.rotation().toQuaternion()
        msg.pose.orientation.x = quat.x()
        msg.pose.orientation.y = quat.y()
        msg.pose.orientation.z = quat.z()
        msg.pose.orientation.w = quat.w()
        self.pose_pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node("gtsam_localizer")
    localizer = GTSAMLocalizer()
    rospy.spin()
"""

OJO! plantea etiquetar los estados en base al
(tiempo)

IMPORTANTE!! El resultado se está acutalizando con la frecuencia del GPs. Hay que preguntarle como
publicar más rápido.

Record a ROS bag:


rosbag record -O test.bag /imu/data /gps/fix /odom/wheel /lidar_odometry



rosbag play test.bag --clock
Ensure timestamps are aligned using:


rostopic echo /imu/data | head -n 10
rostopic echo /gps/fix | head -n 10

You're absolutely right — the assign_pose_index method is currently defined but unused in the code.

Right now, the pose index (self.current_index) is simply incremented (i = self.current_index + 1) rather than being assigned based on the actual message timestamp. This may cause synchronization issues if data doesn't arrive at consistent intervals.

✅ Fix Suggestion: Use assign_pose_index for indexing
We can replace the manual increment with a timestamp-based index like this:

In process_gps, change:

python
Copiar
Editar
i = self.current_index + 1
To:

python
Copiar
Editar
i = self.assign_pose_index(msg.header.stamp)
And make sure self.current_index is updated accordingly after optimization:

python
Copiar
Editar
self.current_index = i
Would you like me to go ahead and apply this fix in the code?


"""
