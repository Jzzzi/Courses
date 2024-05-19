import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import socket
import threading
import time
import math

import multiprocessing as mp

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
        # print("rgb shape", image.shape)
        cv2.imwrite('rgb.jpg',image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low = np.array([40, 40, 0])
        high = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, low, high)
        cv2.imwrite('mask.jpg',mask)
        non_zero_points = cv2.findNonZero(mask)
        # print(non_zero_points.shape)
        if non_zero_points is None or non_zero_points.shape[0] <= 5000:
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
        # self.depth_subscriber = self.create_subscription(Image, '/camera/aligned_depth_to_extcolor/image_raw', self.depth_callback, 10)
        self.camera_info_node = CameraInfoNode()
        self.real_x = 0.
        self.real_y = 0.
        self.depth_img = None
        self.K = None

    def depth_callback(self, msg):
        if self.camera_info_node.camera_info is not None:
            self.K = np.array(self.camera_info_node.camera_info.k).reshape(3, 3)
        else:
            rclpy.spin_once(self.camera_info_node)
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # print(self.depth_img)
        cv2.imwrite('depth.jpg', self.depth_img)
        # print("depth shape", self.depth_img.shape)
    
    def localize_ball_in_3d(self, u, v):
        # TODO: the coordinate of stereo and rgb is different, check the gap
        #       sonetimes return nan, where we should return a memory of previous position
        if self.depth_img is not None and self.K is not None:
            # u,v are the ball central position in rgb image
            # rgb camera info
            fx = 470.0
            fy = 464.0
            cx = 300.0
            cy = 250.0
            # infra1 camera info
            fix = self.K[0, 0]
            fiy = self.K[1, 1]
            cix = self.K[0, 2]
            ciy = self.K[1, 2]
            # convert the rgb position to infra1 position
            u = (u - cx) * fix / fx + cix
            v = (v - cy) * fiy / fy + ciy
            # get the depth value, convert to mm to m
            depth = self.depth_img[int(v), int(u)]*0.001
            # get the angle of the ball
            alpha = math.atan((u - cix) / fix)
            # get the real position of the ball
            real_x = depth * math.cos(alpha)
            real_y = depth * math.sin(alpha) 
            # x axis is the forward direction of the dog head
            # y axis is the left direction of the dog head
            # z axis is the up direction of the dog head
            return self.real_x, self.real_y
        else:
            return 0.5, 0.3
        # self.get_logger().info(f"Ball position relative to camera: x={x}, y={y}, z={z}")

class CameraInfoNode(Node):
    def __init__(self):
        super().__init__('camera_info_node')
        self.info_subscriber = self.create_subscription(CameraInfo, '/camera/depth/camera_info', self.info_callback, 10)
        self.camera_info = None

    def info_callback(self, msg):
        self.camera_info = msg

class BallDetector:
    def __init__(self, ip='10.0.0.144', host='localhost', port=12345):
        self.client_socket = socket.socket()
        self.client_socket.connect((ip, 40000))
        self.rgb = rgbNode()
        self.rsense = RSenseNode()
        
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(3)
        self.x = 0.5
        self.y = -0.2
        self.data = ''
        print(f"Server started at {host}:{port}")
    
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
                # self.make_data(0.5, 0.2, time.time()) # sharp left turn
                self.make_data(0.1, 0., time.time())
            else:
                # TODO: check rpy angle, turn to the corresponding direction
                # self.make_data(0.5, 0.2, time.time())
                self.make_data(0.1, 0., time.time())
        else:
            if True:
                # use RealSense method, which return the accurate relative position
                real_x, real_y = self.rsense.localize_ball_in_3d(x, y)
                self.make_data(np.clip(real_x, 0, 0.5), real_y, time.time())
            else:
                # implemented by Jzzzi, seems nothing wrong
                posi = [float(num) for num in data]
                dist = math.sqrt((posi[0]-posi[2])**2+(posi[1]-posi[3])**2)
                x = x - 320
                fx = 400
                alpha = math.atan(x/fx)
                self.data = f'{np.clip(dist*math.sin(alpha), 0, 1)} {np.clip(dist*math.cos(alpha), -0.5, 0.5)}'

    def handle_client(self, connection, address):
        print(f"Connected by {address}")
        try:
            while True:
                self.get_ball_pos()
                print(self.data)
                data = connection.recv(1024)
                if not data:
                    break  # Connection closed by the client
                print(f"Server received: {data.decode()}")
                connection.sendall(self.data.encode())
        except Exception as e:
            print(f"Error handling connection from {address}: {e}")
        finally:
            connection.close()
            print(f"Connection with {address} closed")
    
    def make_data(self, x, y, time, momentum=0.):
        self.x = self.x * momentum + x * (1 - momentum)
        self.y = self.y * momentum + y * (1 - momentum)
        self.data = f"{self.x} {self.y} {time}"

    def run(self):
        try:
            while True:
                connection, address = self.server_socket.accept()
                thread = threading.Thread(target=self.handle_client, args=(connection, address))
                thread.start()
        except KeyboardInterrupt:
            print("Server is shutting down...")
        finally:
            self.server_socket.close()
            print("Server closed")

    
def main():
    rclpy.init()
    ball_detector = BallDetector()
    ball_detector.run()
    # while True:
    #     ball_detector.get_ball_pos()
    #     print(ball_detector.data)

if __name__ == '__main__':
    main()
    