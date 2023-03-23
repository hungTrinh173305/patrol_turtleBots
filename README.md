#Overview.
- A robot moving freely in 
given spaces, recording 
unknown faces, 
recognizing known ones, 
while avoiding obstacles
- Check "reports" file to see project in details.
#Required packages.
- opencv_python
- ROS Noetic
- face_recognition library from https://github.com/ageitgey/face_recognition. 
- dlib
- numpy
- torch
- torchvision
- typing
- torchstat
- torchsummary
- ptflops
- matplotlib
- onnx
- onnxruntime
#How to run.
1. Create a ros package with image_processing
2. ssh into turtlebot3, then run bringup command: roslaunch turtlebot3_bringup turtlebot3_robot.launch
3. Open another terminal from the turtlebot, run the launch camera command: roslaunch turtlebot3_autorace_camera raspberry_pi_camera_publish.launch
4. Open another terminal of the remote PC, run: 
   - rosrun image_processing detect_with_CascadeClassifier.py --> to run the cv2.CascadeClassifier face recognition module
   - rosrun image_processing detect_with_ultraLight.py        --> to run the ultralight face detection module
   - rosrun image_processing patrol_robot.py                  --> to run the final product with the robot moving and detecting known and unknown faces.


