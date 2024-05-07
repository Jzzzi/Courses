import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,Range

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.subscription = self.create_subscription(
            Image,
            '/image_rgb',
            self.listener_callback,
            10)
        return

    def listener_callback(self, msg):
        self.get_logger().info('Received an image')
        self.get_logger().info('老子得到了一个图噢～')
        print(msg)
        return

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()
    return

if __name__ == '__main__':
    main()
