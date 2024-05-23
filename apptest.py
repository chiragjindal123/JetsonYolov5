import sys
import cv2
import imutils
import time
from yoloDet1 import YoloTRT

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

cap = cv2.VideoCapture(0)

# Set initial signal timings and other variables
green_time = 10  # Default green signal time in seconds
yellow_time = 3  # Default yellow signal time in seconds
red_time = 15  # Default red signal time in seconds
total_signal_time = green_time + yellow_time + red_time

traffic_light_state = "green"  # Initial traffic light state
last_signal_change_time = time.time()  # To track the last time the signal changed

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t, num_vehicles = model.Inference(frame)

    # Display the number of detected vehicles on the frame
    cv2.putText(frame, f"Vehicles Detected: {num_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Adjust signal timings based on the number of detected vehicles
    if num_vehicles <= 1:
        green_time = 15
        yellow_time = 3
        red_time = 10
    elif num_vehicles <= 3:
        green_time = 10
        yellow_time = 2
        red_time = 8
    else:
        green_time = 5
        yellow_time = 2
        red_time = 5

    total_signal_time = green_time + yellow_time + red_time

    # Update the traffic light state based on the adjusted signal timings and time elapsed
    current_time = time.time()
    time_since_last_change = current_time - last_signal_change_time

    if traffic_light_state == "green" and time_since_last_change >= green_time:
        traffic_light_state = "yellow"
        last_signal_change_time = current_time

    elif traffic_light_state == "yellow" and time_since_last_change >= yellow_time:
        traffic_light_state = "red"
        last_signal_change_time = current_time

    elif traffic_light_state == "red" and time_since_last_change >= red_time:
        traffic_light_state = "green"
        last_signal_change_time = current_time

    # Calculate the remaining time for each signal state
    remaining_green_time = max(0, green_time - time_since_last_change)
    remaining_yellow_time = max(0, yellow_time - time_since_last_change)
    remaining_red_time = max(0, red_time - time_since_last_change)

    # Display the traffic light state and timer on the frame
    cv2.putText(frame, "Traffic Light: " + traffic_light_state, (20, 100), 0, 1.5, (0, 255, 255), 3)
    cv2.putText(frame, f"Green: {remaining_green_time:.1f} sec", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Yellow: {remaining_yellow_time:.1f} sec", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Red: {remaining_red_time:.1f} sec", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Output", frame)

    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

