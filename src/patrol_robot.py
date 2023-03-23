#!/usr/bin/env python3
from __future__ import print_function
import sys
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import face_recognition
import dlib

from imutils import face_utils
from box_utils import *
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import time

import rospy,math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class move_and_detect:

  # Instance variables
  def __init__(self):
    self.no_image= 0 # number of unknown faces

    # Load ultralight model
    onnx_path = '/home/trinhh/catkin_ws/src/image_processing/src/UltraLight/models/ultra_light_640.onnx'
    onnx_model = onnx.load(onnx_path)
    predictor = prepare(onnx_model)
    self.ort_session = ort.InferenceSession(onnx_path)
    self.input_name = self.ort_session.get_inputs()[0].name
    self.shape_predictor = dlib.shape_predictor(
      '/home/trinhh/catkin_ws/src/image_processing/src/FacialLandmarks/shape_predictor_5_face_landmarks.dat')

    # Initialize CVBridge
    self.image_pub = rospy.Publisher("new_image_pub", Image, queue_size=10)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback)

    # Add face embedding to database
    tuan_image = face_recognition.load_image_file("/home/trinhh/catkin_ws/src/image_processing/src/tuanTrinh_01.jpg")
    tuan_face_encoding1 = face_recognition.face_encodings(tuan_image)[0]
    self.known_face_encodings = [
      tuan_face_encoding1,
    ]
    self.known_face_names = [
      "Tuan"
    ]
    self.face_locations = []
    self.face_encodings = []
    self.face_names = []
    self.process_this_frame= True

  #Helper function to translate to the correct format
  def xyxy2yxyx(self,boxes):
    new_loc = []
    for i in range(boxes.shape[0]):
      box = boxes[i, :]
      x1, y1, x2, y2 = box
      new_loc.append([y1, x2, y2, x1])
    return new_loc

  #Helper function for callbackMove
  def degrees2radians(self, angle):
    return angle * (math.pi / 180.0)

  def callbackMove(self,data):
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    outData = Twist()

    if data.ranges[0] > 0.3:
        outData.linear.x = 0.35
        outData.angular.z = 0.0
    if data.ranges[0] < 0.3 :
        outData.linear.x = 0
        outData.angular.z = self.degrees2radians(90)
    pub.publish(outData)

  def callback(self,data):
    start1 = time.time()
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # Let's rotate the image by 180 degrees
      cv_image = cv2.rotate(cv_image, cv2.cv2.ROTATE_180)
    except CvBridgeError as e:
      print(e)

    frame = cv_image
    rgb_small_frame = cv_image[:, :, ::-1]

    if frame is not None:
      h, w, _ = frame.shape

      # preprocess img according to the input of UltraLight Model
      img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert bgr to rgb
      img = cv2.resize(img, (640, 480))  # resize
      img_mean = np.array([127, 127, 127])
      img = (img - img_mean) / 128
      img = np.transpose(img, [2, 0, 1])
      img = np.expand_dims(img, axis=0)
      img = img.astype(np.float32)

      #Get the boxes of faces and get their coresponding embeddings
      start = time.time()
      confidences, boxes = self.ort_session.run(None, {self.input_name: img})
      print("Detection time")
      print(time.time()- start)
      boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
      face_locations = self.xyxy2yxyx(boxes)
      start = time.time()
      face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
      print("Encode time")
      print(time.time()- start)

      face_names = []
      for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance =0.45)
        # print(matches)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = self.known_face_names[first_match_index]

        face_names.append(name)

      #Draw boxes on the output
      for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        if face_names[i] == "Unknown": #Check for unknown

          #Save images of unknown
          cv2.imwrite("{}.jpg".format(self.no_image), frame[y1-25 : y2+25 , x1-25: x2+25, :])

          try: #Try to put the unknown embedding to known database
            unknown_encoding = face_recognition.face_encodings(frame[y1-25 : y2+25 , x1-25: x2+25, :])[0]
            print("Success")
            self.known_face_encodings.append(unknown_encoding)
            self.known_face_names.append(str(self.no_image))
            self.no_image += 1
          except:
            x =1

        #Draw rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shape = self.shape_predictor(gray, dlib.rectangle(left=x1, top=y1, right=x2, bottom=y2))
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
          cv2.circle(frame, (x, y), 2, (80, 18, 236), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 18, 236), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80, 18, 236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = face_names[i]
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.3, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    cv2.waitKey(5)
    print("Time all")
    print(time.time() - start1)
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


def main(args):
  robot = move_and_detect()
  sub = rospy.Subscriber('/scan', LaserScan, robot.callbackMove)
  rospy.init_node('move_and_detect', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
