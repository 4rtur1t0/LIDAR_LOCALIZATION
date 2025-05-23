"""
Compute the speed of different topics

"""
# !/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import math
import time

# Subscriptions
ODOMETRY_TOPIC = '/husky_velocity_controller/odom' #, Odometry, self.odom_callback)
ODOMETRY_SCANMATCHING = '/odometry_lidar_scanmatching' #, Odometry, self.odom_sm_callback)
MAP_SCANMATCHING = '/map_sm_global_pose' #, Odometry, self.map_sm_global_pose_callback)
LOCALIZED_POSE = '/localized_pose'# , Odometry, queue_size=10)


class SpeedCalculator:
    def __init__(self):
        # For Odometry
        self.last_odom_time = None
        self.last_odom_pos = None
        self.traversed_distance = 0
        self.timespeed = []
        self.timedistance = []

    def compute_speed(self, msg, topic):
        current_time = msg.header.stamp.to_sec()
        pos = msg.pose.pose.position

        if self.last_odom_time is not None:
            dt = current_time - self.last_odom_time
            if dt > 0:
                dx = pos.x - self.last_odom_pos[0]
                dy = pos.y - self.last_odom_pos[1]
                dz = 0 #pos.z - self.last_odom_pos[2]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                speed = distance / dt
                print(20*'=')
                print(topic)
                print(20 * '=')
                print('Speed is: ', speed)
                rospy.loginfo("Odometry speed: %.3f m/s", speed)
                self.traversed_distance += distance
                self.timespeed.append([current_time, speed])
                self.timedistance.append([current_time, self.traversed_distance])
        self.last_odom_time = current_time
        self.last_odom_pos = (pos.x, pos.y, pos.z)


class NodeSpeedCalculator:
    def __init__(self):
        self.speed_odometry = SpeedCalculator()
        self.speed_sm_odometry = SpeedCalculator()
        self.speed_map_scanmatching = SpeedCalculator()
        self.speed_localized_pose = SpeedCalculator()
        print('Subscribing to odometry topics!')
        # subscribe
        rospy.Subscriber(ODOMETRY_TOPIC, Odometry, self.odom_callback)
        rospy.Subscriber(ODOMETRY_SCANMATCHING, Odometry, self.speed_sm_odometry_callback)
        rospy.Subscriber(MAP_SCANMATCHING, Odometry, self.speed_map_scanmatching_callback)
        rospy.Subscriber(LOCALIZED_POSE, Odometry, self.speed_localized_pose_callback)

    def odom_callback(self, msg):
        self.speed_odometry.compute_speed(msg=msg, topic='Odometry')

    def speed_sm_odometry_callback(self, msg):
        self.speed_odometry.compute_speed(msg=msg, topic='Odometry SM')

    def speed_map_scanmatching_callback(self, msg):
        self.speed_map_scanmatching.compute_speed(msg=msg, topic='Odometry Map SM')

    def speed_localized_pose_callback(self, msg):
        self.speed_localized_pose.compute_speed(msg=msg, topic='Odometry Localized pose')


if __name__ == '__main__':
    rospy.init_node('linear_speed_computer')
    sc = NodeSpeedCalculator()
    rospy.spin()

