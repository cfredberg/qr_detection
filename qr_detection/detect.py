# import sys
# sys.path.append('~/RMRC_venv/lib/python3.10/site-packages')

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from message_filters import Subscriber, TimeSynchronizer

import numpy as np

from pynput import keyboard

import cv2
from pyzbar import pyzbar

import json

from cv_bridge import CvBridge

class QrDetectNode(Node):

    def __init__(self):
        super().__init__('qr_detect')
        self.declare_parameter('use_webcam', False)
        self.declare_parameter('camera_name', "")
        
        use_webcam = self.get_parameter('use_webcam').get_parameter_value().bool_value
        if not use_webcam:
            self.camera_name = self.get_parameter('camera_name').get_parameter_value().string_value
        else:
            self.camera_name = 0

        self.camera_subscription = self.create_subscription(
            Image,
            f'/cameras/raw/camera_{self.camera_name}',
            self.listener_callback,
            1)

        self.collected_codes = []

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()

        self.qr_frame_publisher = self.create_publisher(Image, f'/cameras/qr/camera_{self.camera_name}', 1)
        self.qr_string_publisher = self.create_publisher(String, f'/qr/string/camera_{self.camera_name}', 1)


    def listener_callback(self, frame_msg):
        # Get frames and display them
        frame = self.bridge.imgmsg_to_cv2(frame_msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 30)
        # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        frame = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

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
        
        collected_codes_text = []

        for qr_code in qr_codes:
            (x, y, w, h) = qr_code.rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            text = qr_code.data.decode("utf-8")
            collected_codes_text.append(text)

        new_frame_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.qr_frame_publisher.publish(new_frame_msg)
        
        str_msg = String()
        str_msg.data = collected_codes_text.__str__()
        self.qr_string_publisher.publish(str_msg)

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