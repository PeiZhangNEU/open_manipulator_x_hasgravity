from numpy.core.defchararray import join
import rospy
import numpy as np
# 导入写client需要用到的srv的包
from open_manipulator_msgs.srv import *
from rospy import client

# 导入写subscriber需要用到的msg的包
from open_manipulator_msgs.msg import KinematicsPose
from open_manipulator_msgs.msg import JointPosition
from sensor_msgs.msg import JointState
from ar_track_alvar_msgs.msg import AlvarMarkers
import pybullet as p
import os

rospy.init_node('get_massages_and_control')

data_pose_ar = rospy.wait_for_message('/ar_pose_marker', AlvarMarkers, timeout=None)
ar_pose = [data_pose_ar.markers[0].pose.pose.position.x, data_pose_ar.markers[0].pose.pose.position.y, data_pose_ar.markers[0].pose.pose.position.z]
ar_pose = np.array(ar_pose)
end_orn_quaternion = [data_pose_ar.markers[0].pose.pose.orientation.x, data_pose_ar.markers[0].pose.pose.orientation.y, data_pose_ar.markers[0].pose.pose.orientation.z, data_pose_ar.markers[0].pose.pose.orientation.w]
end_orn = p.getEulerFromQuaternion(end_orn_quaternion)
end_orn = np.array(end_orn)

print(ar_pose)
print(end_orn)