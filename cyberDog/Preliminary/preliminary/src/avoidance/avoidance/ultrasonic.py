import rclpy
import time
from rclpy.node import Node
from protocol.srv import MotionResultCmd
from protocol.msg import MotionServoCmd
from sensor_msgs.msg import Range

SAFE = 0
TURN = 1
STOP = 2

class ultrasonic_sensor(Node):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.sub = self.create_subscription(Range, '/CyberDog_4_1/ultrasonic_payload', self.sub_callback, 10)
        self.status = SAFE
    
    def sub_callback(self, msg: Range):
        self.dist = msg.range
        self.get_logger().info(f"Distance: {self.dist}")
    
def main(args = None):
    rclpy.init(args=args)
    ultra = ultrasonic_sensor("ultrasonic_sensor")
    now = time.time()
    while time.time()-now < 8:
        rclpy.spin_once(ultra, timeout_sec = 1)
    ultra.destroy_node()
    rclpy.shutdown()
