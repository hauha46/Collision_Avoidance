# Collision_Avoidance
The Collision Avoidance program for Drones using Optical Flow
The program use the dronekit package, which includes functions that send the MavLink Message for drones controlling. 
With the Optical Flow, we use the Farneback method to find the flow of the obstacle, draw a bounding box around it
The bounding box which is relative to the frame will tell the drone whether to move left, right or continue the journey. 
After the drone reach the target location, the drone will stop and land.

