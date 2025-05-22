import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from .CharacterDetection import process
from cv_bridge import CvBridge, CvBridgeError
import cv2

class detecton(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.publisher_ = self.create_publisher(Image, 'Detection_Modified', 10)

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.listener_callback,
            10)
        self.subscription
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            modefied_cv_image = process(cv_image)
            #convert the image to ros message
            ros_image = self.bridge.cv2_to_imgmsg(modefied_cv_image, encoding="bgr8")
            self.publisher_.publish(ros_image)
            
            self.get_logger().info("Suucess")
        except CvBridgeError as e:
            self.get_logger().error("CvBridge Error: %s" % str(e))


def main(args=None):
        rclpy.init(args=args)
        detection_node = detecton()
        rclpy.spin(detection_node)
        detection_node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
        main()
