#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

class OdomToTF:
    def __init__(self):
        rospy.init_node('localized_pose_to_tf')

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.static_tf_publisher = tf2_ros.StaticTransformBroadcaster()
        self.odom_sub = rospy.Subscriber('/localized_pose', Odometry, self.odom_callback)
        rospy.Subscriber("/initialpose", PoseWithCovarianceStamped, self.initial_pose_cb)

        self.rate = rospy.Rate(10)

        self.map_tf = TransformStamped()
        self.map_tf.header.stamp = rospy.Time.now()
        self.map_tf.header.frame_id = "odom"  # Typically "odom"
        self.map_tf.child_frame_id = "base_link"    # Typically "base_link"

        # Translation
        self.map_tf.transform.translation.x = 0.0
        self.map_tf.transform.translation.y = 0.0
        self.map_tf.transform.translation.z = 0.0

        # Rotation (as quaternion)
        self.map_tf.transform.rotation.x = 0.0
        self.map_tf.transform.rotation.y = 0.0
        self.map_tf.transform.rotation.z = 0.0
        self.map_tf.transform.rotation.w = 1.0


        self.map_odom_tf = TransformStamped()
        self.map_odom_tf.header.stamp = rospy.Time.now()
        self.map_odom_tf.header.frame_id = "map"  # Typically "odom"
        self.map_odom_tf.child_frame_id = "odom"    # Typically "base_link"

        # Translation
        self.map_odom_tf.transform.translation.x = 0.0
        self.map_odom_tf.transform.translation.y = 0.0
        self.map_odom_tf.transform.translation.z = 0.0

        # Rotation (as quaternion)
        self.map_odom_tf.transform.rotation.x = 0.0
        self.map_odom_tf.transform.rotation.y = 0.0
        self.map_odom_tf.transform.rotation.z = 0.0
        self.map_odom_tf.transform.rotation.w = 1.0
        self.static_tf_publisher.sendTransform(self.map_odom_tf)

    def publish(self):
        self.map_tf.header.stamp = rospy.Time.now()
        self.tf_broadcaster.sendTransform(self.map_tf)

    def sleep(self):
        self.rate.sleep()

    def odom_callback(self, msg):
        print("Entrando al callback")
        # Header
        self.map_tf.header.frame_id = msg.header.frame_id
        self.map_tf.child_frame_id = msg.child_frame_id    

        # Translation
        self.map_tf.transform.translation.x = msg.pose.pose.position.x
        self.map_tf.transform.translation.y = msg.pose.pose.position.y
        self.map_tf.transform.translation.z = msg.pose.pose.position.z

        # Rotation (as quaternion)
        self.map_tf.transform.rotation = msg.pose.pose.orientation

    def initial_pose_cb(self, msg):
        print("Entrando al callback")
        self.map_tf.transform.translation = msg.pose.pose.position
        # Rotation (as quaternion)
        self.map_tf.transform.rotation = msg.pose.pose.orientation



if __name__ == '__main__':
    try:
        node = OdomToTF()

        while not rospy.is_shutdown():
            node.publish()
            node.sleep()

            
    except rospy.ROSInterruptException:
        pass
