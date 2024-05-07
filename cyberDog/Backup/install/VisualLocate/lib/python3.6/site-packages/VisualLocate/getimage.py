import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,Range

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.subscription = self.create_subscription(
            Range,
            '/CyberDog_4_1/ultrasonic_payload',
            self.listener_callback,
            10)
        return

    def listener_callback(self, msg):
        self.get_logger().info('Received an image')
        print(msg)
        return

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin_once(node)
    rclpy.shutdown()
    return

if __name__ == '__main__':
    main()
