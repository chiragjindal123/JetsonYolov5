import sys
import cv2
import imutils
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from yoloDet1 import YoloTRT

# Define the fuzzy input and output variables
vehicle_count = ctrl.Antecedent(np.arange(0, 101, 1), 'vehicle_count')
traffic_light_timings = ctrl.Consequent(np.arange(0, 101, 1), 'traffic_light_timings')

# Define fuzzy membership functions for the input and output variables
vehicle_count['low'] = fuzz.trimf(vehicle_count.universe, [0, 0, 50])
vehicle_count['medium'] = fuzz.trimf(vehicle_count.universe, [0, 50, 100])
vehicle_count['high'] = fuzz.trimf(vehicle_count.universe, [50, 100, 100])

traffic_light_timings['short'] = fuzz.trimf(traffic_light_timings.universe, [0, 0, 50])
traffic_light_timings['medium'] = fuzz.trimf(traffic_light_timings.universe, [0, 50, 100])
traffic_light_timings['long'] = fuzz.trimf(traffic_light_timings.universe, [50, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(vehicle_count['low'], traffic_light_timings['long'])
rule2 = ctrl.Rule(vehicle_count['medium'], traffic_light_timings['medium'])
rule3 = ctrl.Rule(vehicle_count['high'], traffic_light_timings['short'])

# Create the fuzzy control system
traffic_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
traffic_light_timing_sim = ctrl.ControlSystemSimulation(traffic_light_ctrl)

# use path for library and engine file
model = YoloTRT(library="yolov5/build/libmyplugins.so", engine="yolov5/build/yolov5s.engine", conf=0.5, yolo_ver="v5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    detections, t, num_vehicles = model.Inference(frame)

    cv2.putText(frame, f"Vehicles Detected: {num_vehicles}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update the fuzzy input variable with the current vehicle count
    vehicle_count_value = num_vehicles
    traffic_light_timing_sim.input['vehicle_count'] = vehicle_count_value

    # Perform the fuzzy inference
    traffic_light_timing_sim.compute()

    # Get the crisp output value for the traffic light timings
    traffic_light_timings_value = traffic_light_timing_sim.output['traffic_light_timings']

    # Use the crisp output value to adjust the traffic light timings (for example, in seconds)
    # Update your traffic light control logic here based on the traffic_light_timings_value
    # For demonstration purposes, let's assume the traffic light timings are in seconds:
    green_time_seconds = traffic_light_timings_value
    red_time_seconds = 60 - green_time_seconds

    # Print the computed traffic light timings
    print("Traffic Light Timings -")
    print(f"Green Light: {green_time_seconds:.2f} seconds")
    print(f"Red Light: {red_time_seconds:.2f} seconds")

    # Display the output frame in one window
    cv2.imshow("Output", frame)

    # Create an image with the traffic light timer
    timer_image = np.zeros((100, 200, 3), dtype=np.uint8)
    color = (0, 255, 0)  # Green color
    red_time = 60 - green_time_seconds
    text = f"Green: {green_time_seconds:.1f}s  Red: {red_time:.1f}s"
    cv2.putText(timer_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the traffic light timer in another window
    cv2.imshow("Traffic Light Timer", timer_image)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

