import rclpy
from rclpy.node import Node
from protocol.srv import MotionResultCmd
from protocol.msg import MotionServoCmd

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
        self.get_logger().info(f'Publishing: "{msg.motion_id}" "{msg.vel_des}" "{msg.step_height}" {self.count}')
        self.count += 1

def wait_node(node):
    from time import sleep
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
    move_node = move_cmd("move_cmd")
    basic_node.send_request(111)
    wait_node(basic_node)
    import time
    now = time.time()
    while time.time() - now < 3:
        move_node.pub_motion(305, [-0.5, 0., 0.])
    now = time.time()
    while time.time() - now < 1:
        move_node.pub_motion(303, [0., 0., 0.5])
    now = time.time()
    while time.time() - now < 3:
        move_node.pub_motion(303, [0.5, 0., 0.])
    time.sleep(0.5)
    basic_node.send_request(101)
    wait_node(basic_node)
    basic_node.destroy_node()
    move_node.destroy_node()
    rclpy.shutdown()