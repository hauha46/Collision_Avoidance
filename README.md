# Collision_Avoidance
The Collision Avoidance program for Drones using Optical Flow Algorithm

Summary:

The program use OpenCV to analyze and detect obstacle. We also use the dronekit package, which includes functions that send the MavLink Message for drones controlling. 
With the Optical Flow, we use the Farneback method to find the flow of the obstacle,draw a bounding box around it
The bounding box which is relative to the frame will tell the drone whether to move left, right or continue toward the way point. 
After the drone reach the target location, the drone will stop and land.

Sequence of Operations: 

The program will download the waypoint which was given by the Mission Planner software. During the duration of the drone moving toward the waypoint, it will takes input from the camera, apply the Optical Flow algorithm, detect object and then decide which direction to move.

Notes and Flaws: 
- Applying Optical Flow for Collision Avoidance has some flaws. We still need to fine-tune some parameters of the algorithm to get a perfect results. Besides, if the camera moves toward the obstacles, it's hard to filter the obstacles because everything else is moving also, which consequently results a flow for every moving pixel
- The camera also relies on the stabilization of the drone after taking off or during the flight. It also depends on the weather (wind for example)
- This program just apply for the front camera. The drone is absolutely blind to the left and right side.
- With the bounding box idea, it could be apply to avoid objects by going up and down also. 
- However, if we apply Optical Flow for a static camera for object detection, it works perfectly detecting moving object

Reference:

https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html
https://web.archive.org/web/20181007032635/http:/python.dronekit.io/automodule.html#dronekit.CommandSequence
https://github.com/dronekit/dronekit-python/blob/master/examples/guided_set_speed_yaw/guided_set_speed_yaw.py


