import rosbag
import numpy as np
from back_del.quaternion import Quaternion
import sensor_msgs.point_cloud2


class RosbagReader():
    def __init__(self, filename, odo_topic='/odometry', points_topic='/points'):
        self.filename = filename
        self.odo_topic = odo_topic
        self.points_topic = points_topic
        self.odometry = []
        self.pointclouds = []
        # the experiment index to retrieve observations in an order
        self.current_index = 0

    def read_rosbag_data2D(self, deltaxy=0.5, deltath=0.2):
        """
        Read 3D point clouds
        """
        odo = []
        odo_t = []
        scans = []
        scans_times = []
        bag = rosbag.Bag(self.filename)
        print(bag.get_type_and_topic_info())
        # read odometry at deltas of 0.05
        i = 0
        for topic, msg, t in bag.read_messages(topics=[self.odo_topic]):
            # caution using quaternions with real member first!
            q = Quaternion([msg.pose.pose.orientation.w,
                            msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z])
            th = q.Euler()
            if i == 0:
                odo_t.append(t.to_sec())
                odoi = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, th.abg[2]])
                odo.append(odoi)
            odoi1 = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, th.abg[2]])
            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])
            if dxy > deltaxy or dth > deltath:
                odo_t.append(t.to_sec())
                odo.append(odoi1)
                odoi = odoi1
            i += 1
        odo = np.array(odo)
        odo_t = np.array(odo_t)
        print("Read: ", len(odo), "diferent poses from odometry.")

        # read all scans (point clouds and store only the time)
        for topic, msg, t in bag.read_messages(topics=[self.points_topic]):
            scans_times.append(t.to_sec())

        # find the correspondences of odo with scans based on previous times
        scans_times_corr = []
        # find correspondences based on time (closest time to each odometry reading)
        for i in range(len(odo_t)):
            j = np.argmin(np.square(scans_times-odo_t[i]))
            scans_times_corr.append(scans_times[j])

        points = []
        # CAUTION: reading with a particular max and min angle
        for topic, msg, t in bag.read_messages(topics=[self.points_topic]):
            if t.to_sec() in scans_times_corr:
                pp = convert_2dscans(ranges=msg.ranges,
                                     angles=np.linspace(2.3561899662017822, -2.3561899662017822,
                                                        len(msg.ranges)))
                points.append(pp)
        points = np.array(points)
        self.odometry = odo
        self.pointclouds = points
        return odo, points

    def read_rosbag_data3D(self, deltaxy=0.5, deltath=0.2):
        odo = []
        odo_t = []
        bag = rosbag.Bag(self.filename)
        print(bag.get_type_and_topic_info())
        # read odometry at deltas of 0.05
        i = 0
        for topic, msg, t in bag.read_messages(topics=[self.odo_topic]):
            # caution using quaternions with real member first!
            q = Quaternion([msg.pose.pose.orientation.w,
                            msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z])
            th = q.Euler()
            if i == 0:
                odo_t.append(t.to_sec())
                odoi = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, th.abg[2]])
                odo.append(odoi)
            odoi1 = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, th.abg[2]])
            dxy = np.linalg.norm(odoi1[0:2]-odoi[0:2])
            dth = np.linalg.norm(odoi1[2]-odoi[2])
            if dxy > deltaxy or dth > deltath:
                odo_t.append(t.to_sec())
                odo.append(odoi1)
                odoi = odoi1
            i += 1
        odo = np.array(odo)
        odo_t = np.array(odo_t)

        # read the times that correspond to the scans
        scans_times = []
        for topic, msg, t in bag.read_messages(topics=[self.points_topic]):
            scans_times.append(t.to_sec())

        # find the closest times that correspond odo-scans
        scans_times_corr = []
        # find correspondences based on time (closest time to each odometry reading)
        for i in range(len(odo_t)):
            j = np.argmin(np.square(scans_times-odo_t[i]))
            scans_times_corr.append(scans_times[j])

        point_clouds = []
        # get the scans only at those times
        for topic, msg, t in bag.read_messages(topics=[self.points_topic]):
            # if current laser msg time corresponds to one of the correspondences times, add the points
            if t.to_sec() in scans_times_corr:
                point_cloud = []
                for point in sensor_msgs.point_cloud2.read_points(msg, skip_nans=True):
                    point_cloud.append([point[0], point[1], point[2]])
                point_clouds.append(point_cloud)
        point_clouds = np.array(point_clouds)
        self.odometry = odo
        self.pointclouds = point_clouds
        return odo, point_clouds

    def get_observation(self):
        """
        Returns a simulated current observation
        """
        odo = self.odometry[self.current_index]
        points = self.pointclouds[self.current_index]
        self.current_index += 1
        return odo, points


def convert_2dscans(ranges, angles, min_dist=0.5):
    pointcloud = []
    for i in range(len(ranges)):
        if ranges[i] < min_dist:
            continue
        if np.isinf(ranges[i]) or np.isnan(ranges[i]):
            continue
        x = ranges[i] * np.cos(angles[i])
        y = ranges[i] * np.sin(angles[i])
        pointcloud.append(np.array([x, y, 0]))
    pointcloud = np.array(pointcloud)
    return pointcloud
