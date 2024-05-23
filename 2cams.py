import cv2
import imutils
import time
from yoloDet1 import YoloTRT

# Create YoloTRT instances for each camera
model_cam1 = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam2 = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

cap_cam1 = cv2.VideoCapture(0)  # Camera 1
cap_cam2 = cv2.VideoCapture(1)  # Camera 2

# Set initial signal timings and other variables for each direction
green_time = [10, 10]   # Default green signal time in seconds for each direction
yellow_time = [3, 3]   # Default yellow signal time in seconds for each direction
red_time = [15, 15]     # Default red signal time in seconds for each direction
total_signal_time = sum(green_time) + sum(yellow_time) + sum(red_time)

traffic_light_state = ["green", "green"]  # Initial traffic light states for each direction
last_signal_change_time = [time.time()] * 2  # To track the last time the signal changed for each direction

current_camera = 0  # Index of the currently active camera

while True:
    # Switch to the next camera when the current camera shows red light
    if traffic_light_state[current_camera] == "red":
        current_camera = (current_camera + 1) % 2

    # Capture frame from the current camera
    if current_camera == 0:
        ret, frame = cap_cam1.read()
        model = model_cam1
    elif current_camera == 1:
        ret, frame = cap_cam2.read()
        model = model_cam2
    
    if not ret:
        print("Error capturing frame from Camera", current_camera)
        break

    frame = imutils.resize(frame, width=600)

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
    cv2.putText(frame, f"Traffic Light: {traffic_light_state[current_camera]}", (20, 100), 0, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Green: {remaining_green_time:.1f} sec", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Yellow: {remaining_yellow_time:.1f} sec", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Red: {remaining_red_time:.1f} sec", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the number of detected vehicles on the frame for the current camera
    cv2.putText(frame, f"Vehicles Detected: {num_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame for the current camera
    cv2.imshow(f"Camera {current_camera + 1} Output", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap_cam1.release()
cap_cam2.release()
cv2.destroyAllWindows()

