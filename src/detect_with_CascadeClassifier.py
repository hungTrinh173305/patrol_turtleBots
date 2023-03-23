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
import numpy as np
import cv2
import dlib
import numpy as np

class detect_Cascade:

  # instance variables
  def __init__(self):
    self.image_pub = rospy.Publisher("new_image_pub",Image,queue_size=10)
    self.image_sub = rospy.Subscriber("/camera/image",Image,self.callback)
    self.bridge = CvBridge()

    self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    tuan_image = face_recognition.load_image_file("/home/trinhh/catkin_ws/src/image_processing/src/tuanTrinh_01.jpg")
    tuan_face_encoding = face_recognition.face_encodings(tuan_image)[0]

    self.known_face_encodings = [
      tuan_face_encoding
    ]
    self.known_face_names = [
      "Tuan"
    ]
    self.face_locations = []
    self.face_encodings = []
    self.face_names = []
    self.process_this_frame = True

  def xywh2xyxy(self,locs):
    new_loc = []
    for loc in locs:
      x, y, w, h = loc
      new_loc.append([y, x + w, y + h, x])
    return new_loc

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      # Let's rotate the image by 180 degrees
      cv_image = cv2.rotate(cv_image, cv2.cv2.ROTATE_180)
    except CvBridgeError as e:
      print(e)

    small_frame = cv_image
    rgb_small_frame = small_frame[:, :, ::-1]

    if True:
      gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      face_locations = self.face_cascade.detectMultiScale(gray, 1.3, 5)
      encoded_locations = self.xywh2xyxy(face_locations)
      face_encodings = face_recognition.face_encodings(rgb_small_frame, encoded_locations)
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

      for (x, y, w, h), name in zip(face_locations, face_names):
        cv_image = cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = cv_image[y:y + h, x:x + w]
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(cv_image, name, (x + 6, y - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)


def main(args):
  robot = detect_Cascade()
  rospy.init_node('detect_Cascade', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
