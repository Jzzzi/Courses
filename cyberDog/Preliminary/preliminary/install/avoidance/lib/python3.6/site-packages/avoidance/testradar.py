import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from sensor_msgs.msg import LaserScan

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
        self.get_logger().info(f"Radar tested okay")
        self.get_logger().info(f"the distance to each direction is {msg.ranges}")
        
def main(args=None):
    rclpy.init(args=args)
    radar = radar_sensor("radar_sensor")
    rclpy.spin_once(radar, timeout_sec=0.12)
    radar.destroy_node()
    rclpy.shutdown()