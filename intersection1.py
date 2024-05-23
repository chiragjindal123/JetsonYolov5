import sys
import cv2
import imutils
import time
from yoloDet1 import YoloTRT

def initialize_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error opening Camera {camera_index}")
        sys.exit(1)
    return cap

def release_camera(cap):
    if cap.isOpened():
        cap.release()

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error capturing frame")
    return frame

# Create YoloTRT instances for each camera
model_cam1 = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam2 = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam3 = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

# Initialize cameras
cap_cam1 = initialize_camera(0)  # Camera 1
cap_cam2 = initialize_camera(1)  # Camera 2
cap_cam3 = initialize_camera(2)  # Camera 3

# Set initial signal timings and other variables for each direction
green_time = [10, 10, 10]   # Default green signal time in seconds for each direction
yellow_time = [3, 3, 3]   # Default yellow signal time in seconds for each direction
red_time = [15, 15, 15]     # Default red signal time in seconds for each direction
total_signal_time = sum(green_time) + sum(yellow_time) + sum(red_time)

traffic_light_state = ["green", "green", "green"]  # Initial traffic light states for each direction
last_signal_change_time = [time.time()] * 3  # To track the last time the signal changed for each direction

current_camera = 0  # Index of the currently active camera

# Reduced frame resolution and frames per second (fps)
resolution = (160, 120)
fps = 15

while True:
    # Switch to the next camera when the current camera shows red light
    if traffic_light_state[current_camera] == "red":
        if current_camera == 0:
            cap_cam1.release()
        elif current_camera == 1:
            cap_cam2.release()
        elif current_camera == 2:
            cap_cam3.release()

        current_camera = (current_camera + 1) % 3

        if current_camera == 0:
            cap_cam1 = cv2.VideoCapture(0)
        elif current_camera == 1:
            cap_cam2 = cv2.VideoCapture(1)
        elif current_camera == 2:
            cap_cam3 = cv2.VideoCapture(2)

    # Capture frame from the current camera
    if current_camera == 0:
        ret, frame = cap_cam1.read()
        model = model_cam1
    elif current_camera == 1:
        ret, frame = cap_cam2.read()
        model = model_cam2
    elif current_camera == 2:
        ret, frame = cap_cam3.read()
        model = model_cam3

    if not ret:
        print("Error capturing frame from Camera", current_camera)
        break

    frame = imutils.resize(frame, width=resolution[0])

    # Perform inference and traffic light logic for the current camera
    detections, t, num_vehicles = model.Inference(frame)

    # Adjust signal timings based on the number of detected vehicles
    if num_vehicles <= 1:
        green_time[current_camera] = 15
        yellow_time[current_camera] = 3
        red_time[current_camera] = 10
    elif num_vehicles <= 3:
        green_time[current_camera] = 10
        yellow_time[current_camera] = 2
        red_time[current_camera] = 8
    else:
        green_time[current_camera] = 5
        yellow_time[current_camera] = 2
        red_time[current_camera] = 5

    total_signal_time = sum(green_time) + sum(yellow_time) + sum(red_time)

    # Update the traffic light state based on the adjusted signal timings and time elapsed
    current_time = time.time()
    time_since_last_change = current_time - last_signal_change_time[current_camera]

    if traffic_light_state[current_camera] == "green" and time_since_last_change >= green_time[current_camera]:
        traffic_light_state[current_camera] = "yellow"
        last_signal_change_time[current_camera] = current_time

    elif traffic_light_state[current_camera] == "yellow" and time_since_last_change >= yellow_time[current_camera]:
        traffic_light_state[current_camera] = "red"
        last_signal_change_time[current_camera] = current_time

    elif traffic_light_state[current_camera] == "red" and time_since_last_change >= red_time[current_camera]:
        traffic_light_state[current_camera] = "green"
        last_signal_change_time[current_camera] = current_time

    # Calculate the remaining time for each signal state
    remaining_green_time = max(0, green_time[current_camera] - time_since_last_change)
    remaining_yellow_time = max(0, yellow_time[current_camera] - time_since_last_change)
    remaining_red_time = max(0, red_time[current_camera] - time_since_last_change)

    # Display the traffic light state and timer on the frame for the current camera
    cv2.putText(frame, f"Traffic Light: {traffic_light_state[current_camera]}", (20, 40), 0, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Green: {remaining_green_time:.1f} sec", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Yellow: {remaining_yellow_time:.1f} sec", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Red: {remaining_red_time:.1f} sec", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Vehicles Detected: {num_vehicles}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame for the current camera
    cv2.imshow(f"Camera {current_camera + 1} Output", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(int(1000 / fps))
    if key == ord('q'):
        break

cap_cam1.release()
cap_cam2.release()
cap_cam3.release()
cv2.destroyAllWindows()

