import cv2 as cv
import numpy as np
import dronekit as dk
from pymavlink import mavutil
import time
cap = cv.VideoCapture(0)
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
connection_string = "127.0.0.1:14550"
vehicle = dk.connect(connection_string, wait_ready=True)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi',fourcc, 20.0, (640,480))

def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = dk.VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)




def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def goto_position_target_local_ned(north, east, down):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        north, east, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0,  # x, y, z velocity in m/s  (not used)
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)
    vehicle.flush()

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)

    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        # if ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) ** 0.5 > 15:
        if ((x2 - x1) > 10 or (x2 - x1) < -10) and ((y2 - y1) > 10 or (y2 - y1) < -10):
            cv.circle(vis, (x1, y1), 15, (0, 0, 255), -1)
            cv.circle(vis, (x2, y2), 15, (0, 0, 255), -1)

    # cv.polylines(vis, lines, 0, (0, 255, 0))

    return vis
cmds = vehicle.commands
cmds.download()
cmds.wait_ready()

# init_lat = cmds[1].x #34.0531515
# init_lon = cmds[1].y #-114.67867319999999
waypoint = dk.LocationGlobalRelative(cmds[1].x, cmds[1].y, 10)  # Destination
arm_and_takeoff(2)
vehicle.simple_goto(waypoint)
time.sleep(5)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 5, 15, 3, 5, 1.2, 0)
    new_frame = draw_flow(frame_gray, flow)
    frame_HSV = cv.cvtColor(new_frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (0, 58, 140), (57, 255, 255))
    ret, thresh = cv.threshold(frame_threshold, 50, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > 5000 and area < 13000:
            contours[0] = contours[i]
            x, y, w, h = cv.boundingRect(contours[i])
            cv.rectangle(frame_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_x = x + w / 2
            center_y = y + h / 2
            if (center_x < 160):
                if (x + w <= 160):  # when the obstacle is on the left side of the screen < 160
                    print("go straight")
                else:
                    print("turn left")  # when the obstacle is on the left side but it's > 160
                    # vehicle.mode = dk.VehicleMode("GUIDED")
                    send_ned_velocity(0, 0, 0)  # stop the vehicle for 5 seconds
                    time.sleep(2)
                    goto_position_target_local_ned(0, -0.5, 0)  # move left for 0.5 meters
                    time.sleep(3)
                    vehicle.simple_goto(waypoint)  # continue the journey
            elif (center_x > 160) and (center_x <= 320):
                print("turn left")  # the obstacle is on the center_left of the screen
                # vehicle.mode = dk.VehicleMode("GUIDED")
                send_ned_velocity(0, 0, 0)
                time.sleep(2)
                goto_position_target_local_ned(0, -0.5, 0)
                time.sleep(3)
                vehicle.simple_goto(waypoint)
            elif (center_x > 320) and (center_x < 480):  # the obstacle is on the center_right of the screen
                # vehicle.mode = dk.VehicleMode("GUIDED")
                print("turn right")
                send_ned_velocity(0, 0, 0)  # stop the vehicle for 5 seconds
                time.sleep(2)
                goto_position_target_local_ned(0, 0.5, 0)  # move left for 0.5 meters
                time.sleep(3)
                vehicle.simple_goto(waypoint)  # continue the journey
            elif (center_x >= 480):  # the obstacle is on the right of the screen
                if (x < 480):
                    print("turn right")
                    # vehicle.mode = dk.VehicleMode("GUIDED")
                    send_ned_velocity(0, 0, 0)
                    time.sleep(2)
                    goto_position_target_local_ned(0, 0.5, 0)
                    time.sleep(3)
                    vehicle.simple_goto(waypoint)
                else:
                    print("go straight")
    save = cv.cvtColor(frame_gray, cv.COLOR_GRAY2BGR)
    out.write(save)
    cv.imshow("OpticalFlow", new_frame)
    cv.imshow("Original", frame_gray)
    old_gray = frame_gray.copy()

    key = cv.waitKey(30)
    if key == ord('q'):
        out.release()
        break
    lat = vehicle.location.global_relative_frame.lat  # get the current latitude
    lon = vehicle.location.global_relative_frame.lon  # get the current longitude
    if lat == cmds[1].x and lon == cmds[1].y:  # check whether the vehicle is arrived or not
        print("Arrived")
        out.release()
        break
print("Landing")
vehicle.mode = dk.VehicleMode("LAND")
vehicle.flush()
