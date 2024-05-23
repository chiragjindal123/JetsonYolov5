import cv2
import imutils
import time
import numpy as np
from yoloDet1 import YoloTRT

# Define the polygon points for the left halves of both cameras
polygon_points = np.array([
    [(24, 165), (70, 153), (274, 298), (74, 298)]
])

# Create separate YoloTRT instances for each half of the cameras
model_cam1_left = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam1_right = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam2_left = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")
model_cam2_right = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

cap_cam1 = cv2.VideoCapture(0)  # Camera 1
cap_cam2 = cv2.VideoCapture(1)  # Camera 2

# Initial signal timings and other variables for each direction
green_time = [10, 10]   # Default green signal time in seconds for each direction
yellow_time = [3, 3]   # Default yellow signal time in seconds for each direction
red_time = [15, 15]     # Default red signal time in seconds for each direction

# Initial traffic light states for each direction
traffic_light_state = ["green", "red"]  # Start first camera with green and second with red
last_signal_change_time = [time.time(), time.time()]  # Initialize last signal change time for both cameras

total_signal_time = sum(green_time) + sum(yellow_time) + sum(red_time)

current_camera = 0  # Index of the currently active camera
current_half = "left"  # Indicate the currently active half for the current camera



while True:
    # Capture frame from the current camera
    if current_camera == 0:
        ret, frame = cap_cam1.read()
        if current_half == "left":
            model = model_cam1_left
        else:
            model = model_cam1_right
    elif current_camera == 1:
        ret, frame = cap_cam2.read()
        if current_half == "left":
            model = model_cam2_left
        else:
            model = model_cam2_right
    
    if not ret:
        print("Error capturing frame from Camera", current_camera)
        break

    frame = imutils.resize(frame, width=800)  # Reduce the width for smaller windows

    # Split the frame into left and right halves
    width = frame.shape[1] // 2
    left_half = frame[:, :width]
    right_half = frame[:, width:]

    # Draw the polygon on the processing frame
    cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Choose the current processing half based on the current state
    if current_half == "left":
        processing_frame = left_half
        mask = np.zeros_like(processing_frame)
        cv2.fillPoly(mask, polygon_points, (255, 255, 255))  # Create a polygon mask
        masked_frame = cv2.bitwise_and(processing_frame, mask)  # Apply the mask
    else:
        processing_frame = right_half

    # Perform inference and traffic light logic for the masked processing frame
    detections, t, num_vehicles = model.Inference(masked_frame)

    # Adjust signal timings based on the number of detected vehicles
    if num_vehicles <= 1:
        green_time[current_camera] = 15
        yellow_time[current_camera] = 3
        red_time[current_camera] = 2
    elif num_vehicles <= 3:
        green_time[current_camera] = 10
        yellow_time[current_camera] = 2
        red_time[current_camera] = 2
    else:
        green_time[current_camera] = 5
        yellow_time[current_camera] = 2
        red_time[current_camera] = 2

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
        if current_half == "left":
            current_half = "right"
        else:
            current_half = "left"
            current_camera = (current_camera + 1) % 2
        traffic_light_state[current_camera] = "green"
        last_signal_change_time[current_camera] = current_time

    # Calculate the remaining time for each signal state
    remaining_green_time = max(0, green_time[current_camera] - time_since_last_change)
    remaining_yellow_time = max(0, yellow_time[current_camera] - time_since_last_change)
    remaining_red_time = max(0, red_time[current_camera] - time_since_last_change)

    # Display the traffic light state and timer on the processing frame
    cv2.putText(processing_frame, f"Traffic Light: {traffic_light_state[current_camera]}", (20, 50), 0, 1, (0, 255, 255), 2)
    cv2.putText(processing_frame, f"Green: {remaining_green_time:.1f} sec", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(processing_frame, f"Yellow: {remaining_yellow_time:.1f} sec", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(processing_frame, f"Red: {remaining_red_time:.1f} sec", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(processing_frame, f"Vehicles Detected: {num_vehicles}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the processing frame for the current half of the camera
    cv2.imshow(f"Camera {current_camera + 1} {current_half.capitalize()} Half Output", processing_frame)

    
    
    # Check for key press to exit the loop
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release resources and close windows
cap_cam1.release()
cap_cam2.release()
cv2.destroyAllWindows()