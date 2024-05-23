import cv2
import time

def switch_cameras(current_camera):
    current_camera.release()
    next_camera_index = (current_camera_index + 1) % len(camera_indices)
    next_camera = cv2.VideoCapture(camera_indices[next_camera_index])
    return next_camera, next_camera_index

# List of camera indices
camera_indices = [0, 1, 2]  # Update with your camera indices

current_camera_index = 0
current_camera = cv2.VideoCapture(camera_indices[current_camera_index])

start_time = time.time()

while True:
    ret, frame = current_camera.read()

    if not ret:
        break

    cv2.imshow('Current Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Check if it's time to switch cameras
    if time.time() - start_time >= 10:
        current_camera.release()  # Release the paused camera
        current_camera, current_camera_index = switch_cameras(current_camera)
        start_time = time.time()

cv2.destroyAllWindows()
current_camera.release()

