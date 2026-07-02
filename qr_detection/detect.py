# import sys
# sys.path.append('~/RMRC_venv/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from message_filters import Subscriber, TimeSynchronizer

import numpy as np

from pynput import keyboard

import cv2
from pyzbar import pyzbar

import json

from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, ReliabilityPolicy

from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import PoseStamped

class QrDetectNode(Node):

    def __init__(self):
        super().__init__('qr_detect')
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('camera_id', 0)
        
        use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        if not use_webcam:
            self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        else:
            self.camera_id = 0

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.camera_subscription = self.create_subscription(
            CompressedImage,
            f'/cameras/raw/camera_{self.camera_id}',
            self.listener_callback,
            qos)

        self.collected_codes = []

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()

        self.qr_frame_publisher = self.create_publisher(CompressedImage, f'/cameras/qr/camera_{self.camera_id}', qos)
        self.qr_string_publisher = self.create_publisher(String, f'/qr/string/camera_{self.camera_id}', 1)

        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(1.0, self.publish_marker)


    def listener_callback(self, frame_msg):
        # Get frames and display them
        frame = self.bridge.compressed_imgmsg_to_cv2(frame_msg, desired_encoding='bgr8')
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 30)
        # # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        # data, bbox, _ = self.qr_detector.detectAndDecode(frame)
        # print(f"data:\n{data}\nbbox data type:\n{type(bbox)}\nBounding box: {bbox}")
        # if not isinstance(bbox, type(None)):
        #     point_0 = (int(bbox[0][0][0]), int(bbox[0][0][1]))
        #     point_1 = (int(bbox[0][1][0]), int(bbox[0][1][1]))
        #     point_2 = (int(bbox[0][2][0]), int(bbox[0][2][1]))
        #     point_3 = (int(bbox[0][3][0]), int(bbox[0][3][1]))
        #     color = (0, 255, 0)
        #     thinkness = 2
        #     cv2.line(frame, point_0, point_1, color, thinkness)
        #     cv2.line(frame, point_1, point_2, color, thinkness)
        #     cv2.line(frame, point_2, point_3, color, thinkness)
        #     cv2.line(frame, point_3, point_0, color, thinkness)

        qr_codes = pyzbar.decode(frame)

        for qr_code in qr_codes:
            (x, y, w, h) = qr_code.rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            text = qr_code.data.decode("utf-8")
            if text not in self.collected_codes:
                self.collected_codes.append(text)
                self.publish_marker()

        new_frame_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpg')
        new_frame_msg.header.stamp = self.get_clock().now().to_msg()
        self.qr_frame_publisher.publish(new_frame_msg)
        
        str_msg = String()
        str_msg.data = self.collected_codes.__str__()
        self.qr_string_publisher.publish(str_msg)

    def publish_marker(self):
        try:
            # Step 1: lookup map→odom
            map_to_odom = self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
            )

            # Step 2: lookup odom→base_link
            odom_to_base = self.tf_buffer.lookup_transform(
                "odom", "base_link", rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0)
            )

            # Step 3: compose transforms manually
            pose = PoseStamped()
            pose.header.frame_id = "base_link"
            pose.pose.orientation.w = 1.0  # identity orientation

            # transform into odom
            pose_in_odom = do_transform_pose(pose, odom_to_base)
            # transform into map
            pose_in_map = do_transform_pose(pose_in_odom, map_to_odom)

            x = pose_in_map.pose.position.x
            y = pose_in_map.pose.position.y

            # Step 4: publish marker at robot position in map frame
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.3
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0  # green marker
            self.publisher.publish(marker)

        except Exception as e:
            self.get_logger().warn(f"Could not chain transforms: {e}")

def main(args=None):
    rclpy.init(args=args)

    qr_detect_node = QrDetectNode()

    rclpy.spin(qr_detect_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    qr_detect_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()