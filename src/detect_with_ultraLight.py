#!/usr/bin/env python3
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import face_recognition
import numpy as np
import cv2
import dlib
from imutils import face_utils
from box_utils import *

import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare


class detect_ultralight:

    # instance variables
    def __init__(self):
        onnx_path = '/home/trinhh/catkin_ws/src/image_processing/src/UltraLight/models/ultra_light_640.onnx'
        onnx_model = onnx.load(onnx_path)
        predictor = prepare(onnx_model)
        self.ort_session = ort.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        self.shape_predictor = dlib.shape_predictor(
            '/home/trinhh/catkin_ws/src/image_processing/src/FacialLandmarks/shape_predictor_5_face_landmarks.dat')

        self.image_pub = rospy.Publisher("new_image_pub", Image, queue_size=10)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # print(cv_image.shape)

            # Let's rotate the image by 180 degrees
            cv_image = cv2.rotate(cv_image, cv2.cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)


        frame = cv_image

        if frame is not None:
            h, w, _ = frame.shape

            # preprocess img acquired
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
            img = cv2.resize(img, (640, 480))  # resize
            img_mean = np.array([127, 127, 127])
            img = (img - img_mean) / 128
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            confidences, boxes = self.ort_session.run(None, {self.input_name: img})
            boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                x1, y1, x2, y2 = box
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                shape = self.shape_predictor(gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 2, (80, 18, 236), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
                cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                text = f"Face: {round(probs[i], 2)}"
                cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

        cv2.imshow("Image window", frame)
        cv2.waitKey(3)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            print(e)


def main(args):
    robot = detect_ultralight()
    rospy.init_node('detect_ultralight', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
