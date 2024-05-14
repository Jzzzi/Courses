import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import socket
import time
import math

class rgbNode(Node):
    def __init__(self):
        super().__init__('rgb_node')
        self.subscription = self.create_subscription(
            Image,
            '/image_rgb',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        self.avgx = 320
        self.avgy = 240
        self.in_sight_flag = False

    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        image = self.bridge.imgmsg_to_cv2(msg)
        print(image.shape)
        # cv2.imwrite('rgb.jpg',image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = np.array([40, 40, 0])
        high = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, low, high)
        # cv2.imwrite('mask.jpg',mask)
        non_zero_points = cv2.findNonZero(mask)
        print(non_zero_points.shape)
        if non_zero_points is None:
            self.in_sight_flag = False
            print("ball not in sight!")
        else:
            self.in_sight_flag = True
            x_coords = [point[0][0] for point in non_zero_points]
            y_coords = [point[0][1] for point in non_zero_points]
            self.avgx = np.mean(x_coords)
            self.avgy = np.mean(y_coords)
        # print(f'Average position: ({avg_x}, {avg_y})')
        # print(len(non_zero_points))

    def get_avg(self):
        return self.avgx,self.avgy


class RSenseNode(Node):
    def __init__(self):
        super().__init__('real_sense_node')
        self.bridge = CvBridge()
        self.depth_subscriber = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.info_subscriber = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_callback, 10)
        self.camera_info = None
        self.depth_img = None
        self.K = None

    def info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        if self.camera_info is not None:
            self.K = np.array(self.camera_info.k).reshape(3, 3)
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    ''' 
    def localize_ball_in_3d(self, u, v, depth, K):
        focal_length = K[0, 0]
        principal_point = (K[0, 2], K[1, 2])
        x = (u - principal_point[0]) * depth / focal_length
        y = (v - principal_point[1]) * depth / focal_length
        z = depth
        self.x, self.y, self.z = x, y, z
        self.get_logger().info(f"Ball position relative to camera: x={x}, y={y}, z={z}")
    '''
    def localize_ball_in_3d(self, u, v):
        # TODO: the coordinate of stereo and rgb is different, check the gap
        #       sonetimes return nan, where we should return a memory of previous position
        if self.depth_img is not None and self.K is not None:
            depth = self.depth_img[int(v), int(u)]
            focal_length = self.K[0, 0]
            principal_point = (self.K[0, 2], self.K[1, 2])
            x = (u - principal_point[0]) * depth / focal_length
            y = (v - principal_point[1]) * depth / focal_length
            z = depth
            alpha = math.atan(x/z)
            real_x = z * math.sin(alpha)
            real_y = z * math.cos(alpha)
            return real_x, real_y
        else:
            return 0.5, -0.2
        # self.get_logger().info(f"Ball position relative to camera: x={x}, y={y}, z={z}")

class BallDetector:
    def __init__(self, ip):
        self.client_socket = socket.socket()
        self.client_socket.connect((ip, 40000))
        self.rgb = rgbNode()
        self.rsense = RSenseNode()
    
    def refresh(self):
        pass
    
    def get_ball_pos(self):
        # check coordinate frame and axis order
        msg = 'start'
        self.client_socket.send(msg.encode())
        data = self.client_socket.recv(1024).decode()
        data = data.split(' ')
        rclpy.spin_once(self.rgb)
        rclpy.spin_once(self.rsense)
        x, y = self.rgb.get_avg()
        if self.rgb.in_sight_flag == False:
            if data is None:
                print("Man! What can I say?")
                return np.array([0.5, 0.2, 0.]) # sharp left turn
            else:
                # TODO: check rpy angle, turn to the corresponding direction
                return np.array([0.5, 0.2, 0.])
        else:
            if data is None:
                # use RealSense method, which return the accurate relative position
                real_x, real_y = self.rsense.localize_ball_in_3d(x, y)
                return np.array([real_x, real_y, 0.])
            else:
                # implemented by Jzzzi, seems nothing wrong
                posi = [float(num) for num in data]
                dist = math.sqrt((posi[0]-posi[2])**2+(posi[1]-posi[3])**2)
                x = x - 320
                fx = 400
                alpha = math.atan(x/fx)
                return np.array([dist*math.sin(alpha), dist*math.cos(alpha), 0.])
                

def main():
    rclpy.init()
    ip = '10.0.0.144'
    ball_detector = BallDetector(ip)
    while True:
        print(ball_detector.get_ball_pos())
        # ball_detector.get_ball_pos()
        time.sleep(0.1)

if __name__ == '__main__':
    main()
    