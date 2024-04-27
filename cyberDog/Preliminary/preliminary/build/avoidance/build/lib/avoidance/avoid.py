import rclpy
from rclpy.node import Node
from protocol.srv import MotionResultCmd
from protocol.msg import MotionServoCmd
from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan

SAFE = 0
TURN = 1
STOP = 2

SMOOTH_DIR = 8
RIGHT_HALF = 3
LEFT_HALF = 4

class basic_cmd(Node):
    def __init__(self, name):
        super().__init__(name)
        self.client = self.create_client(MotionResultCmd, '/CyberDog_4_1/motion_result_cmd')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.request = MotionResultCmd.Request()
    
    def send_request(self, motion_id):
        self.request.motion_id = motion_id
        self.future = self.client.call_async(self.request)

class move_cmd(Node):
    def __init__(self, name):
        super().__init__(name)
        self.publisher_ = self.create_publisher(MotionServoCmd, '/CyberDog_4_1/motion_servo_cmd', 10)
        self.count = 0
    
    def pub_motion(self, motion_id, vel_des, step_height=[0.05, 0.05]):
        msg = MotionServoCmd()
        msg.motion_id = motion_id
        msg.vel_des = vel_des
        msg.step_height = step_height
        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg.motion_id}" "{msg.vel_des}" "{msg.step_height}" {self.count}')
        self.count += 1

class ultrasonic_sensor(Node):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.sub = self.create_subscription(Range, '/CyberDog_4_1/ultrasonic_payload', self.sub_callback, 10)
        self.status = SAFE
    
    def sub_callback(self, msg: Range):
        self.dist = msg.range
        if self.dist <= 0.6:
            self.status = TURN
            self.get_logger().info(f"Need to turn")
        elif self.dist <= 0.3:
            self.status = STOP
            self.get_logger().info(f"Need to stop")
        else:
            self.status = SAFE
    
    def get_status(self):
        return self.status

class radar_sensor(Node):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.sub = self.create_subscription(LaserScan, '/CyberDog_4_1/scan', self.sub_callback, 10)
        self.direction = 0
    
    def sub_callback(self, msg: LaserScan):
        self.ranges = [float("inf") for _ in range(SMOOTH_DIR)]
        len = msg.ranges.__len__()
        for i in range(len):
            self.ranges[i * SMOOTH_DIR // len] = self.min(self.ranges[i * SMOOTH_DIR // len], msg.ranges[i]) 
        # self.ranges[17] = self.ranges[17] * 18. / ((len - 1) % 18 + 1)
        self.direction  = self.ranges.index(max(self.ranges))
        self.get_logger().info(f"the distance to each direction is {self.ranges}")
        
    def min(self, a, b):
        if a == 0.0:
            return b
        elif b == 0.0:
            return a
        else:
            return a if a < b else b
    
    def get_direction(self):
        return self.direction
    
    def direction_ok(self):
        return (self.direction >= RIGHT_HALF and self.direction <= LEFT_HALF)


def wait_node(node):
    while rclpy.ok():
        rclpy.spin_once(node)
        if node.future.done():
            try:
                response = node.future.result()
            except Exception as e:
                node.get_logger().info('Service call failed %r'%(e,))
            else:
                node.get_logger().info("cmd has done!")
            break
    
def main(args=None):
    rclpy.init(args=args)
    basic_node = basic_cmd("basic_cmd")
    move = move_cmd("move_cmd")
    ultra = ultrasonic_sensor("ultrasonic_sensor")
    radar = radar_sensor("radar_sensor")
    
    basic_node.send_request(111)
    wait_node(basic_node)
    
    from time import time
    now = time()
    while time() - now < 30:
        rclpy.spin_once(ultra, timeout_sec=0.12)
        if ultra.get_status() == SAFE:
            move.pub_motion(303, [0.3, 0., 0.])
        elif ultra.get_status() == TURN:
            rclpy.spin_once(radar, timeout_sec=0.12)
            if radar.get_direction() <= RIGHT_HALF:
                move.pub_motion(303, [0., 0., 0.1])
                move.get_logger().info("Turning Left")
            else:
                move.pub_motion(303, [0., 0., -0.1])
                move.get_logger().info("Turning Right")
            rclpy.spin_once(radar, timeout_sec=0.12)
        elif ultra.get_status() == STOP:
            move.pub_motion(303, [-0.3, 0., 0.])
            
    # now = time.time()
    # while time.time() - now < 3:
    #     move_node.pub_motion(305, [-0.5, 0., 0.])
    # now = time.time()
    # while time.time() - now < 1:
    #     move_node.pub_motion(303, [0., 0., 0.5])
    # now = time.time()
    # while time.time() - now < 3:
    #     move_node.pub_motion(303, [0.5, 0., 0.])
    # time.sleep(0.5)
    
    basic_node.send_request(101)
    wait_node(basic_node)
    
    basic_node.destroy_node()
    move.destroy_node()
    ultra.destroy_node()
    radar.destroy_node()
    rclpy.shutdown()