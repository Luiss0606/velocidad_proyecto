from collections import defaultdict

import cv2
import numpy as np
import time
import math
import pickle
from sklearn.preprocessing import PolynomialFeatures
from collections import deque


from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')
# https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
classes_dict = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    }
classes_list=list(classes_dict.keys())


# Open the video file
video_path = "./videos_prueba/1.mp4"
cap = cv2.VideoCapture(video_path)

# Create a video writer for the output video
output_path = "./runs/1_annotated.mp4"
codec = cv2.VideoWriter_fourcc(*"XVID")  # You can change the codec as needed
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, codec, fps, frame_size)



line = [(555, 565), (1073, 565)]
count_up = 0
count_down = 0

# Store the track history
track_history = defaultdict(lambda: [])
speed_data = defaultdict(lambda: [])
object_couter = defaultdict(lambda: [])

def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def get_direction(last_pos, first_pos):
    direction = ""

    if first_pos[1] < last_pos[1]:
        direction += "Shout"
    elif first_pos[1] > last_pos[1]:
        direction += "North"
    else:
        direction += ""
    return direction


def ppm_calculate(x_pos, y_pos):
    with open('./pruebas/ppm_v3.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.predict(PolynomialFeatures(2).fit_transform([[x_pos, y_pos]]))[0]


def estimate_speed(first_pos, last_pos, first_t, last_t):
    d_pixels = math.sqrt(math.pow(last_pos[0] - first_pos[0], 2) + math.pow(last_pos[1] - first_pos[1], 2))
    middle_point = (int((last_pos[0] + first_pos[0]) / 2), int((last_pos[1] + first_pos[1]) / 2))
    ppm = ppm_calculate(middle_point[0], middle_point[1])
    d_meters = d_pixels / ppm
    dt = first_t - last_t  # in seconds
    speed = d_meters / dt  # in m/s
    speed = speed * 3.6  # convert to km/h
    return int(speed)



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame, 
            persist=True,
            conf=0.7,
            iou=0.8,
            device=0,
            tracker='bytetrack.yaml',
            classes=classes_list,)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh
        try :
            track_ids = results[0].boxes.id.int().tolist()
        except:
            pass

        print(track_ids)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot(
            line_width=2,  # thickness of bounding box and track line
            font_size=2,  # font size of track ID
        )
        # Draw the line in the middle of the frame
        cv2.line(annotated_frame, (0, int(annotated_frame.shape[0] / 2)), (annotated_frame.shape[1], int(annotated_frame.shape[0] / 2)), (46, 162, 112), 3)

        
        # cv2.line(annotated_frame, (0, int(737.94654122)), (annotated_frame.shape[1], int(-0.22085729 * annotated_frame.shape[1] + 737.94654122)), (0, 0, 255), 3)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            box_xc, box_yc, box_w, box_h = box

            # --------Tracking--------
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=30)
                speed_data[track_id] = []

            center_bottom = (int(box_xc), int(box_yc + box_h / 2))
            track_history[track_id].appendleft((center_bottom, time.time()))

            current_track = track_history[track_id]
            current_track_pos = [x[0] for x in current_track]
            current_track_time = [x[1] for x in current_track]

            # Draw the track
            for i in range(1,len(current_track_pos)):
                    pos_x, pos_y = current_track_pos[i][0], current_track_pos[i][1]
                    thickness = int(np.sqrt(64 / float(i+i)) * 1.5)
                    pos_x1, pos_y1 = current_track_pos[i-1][0], current_track_pos[i-1][1]
                    cv2.line(annotated_frame, (int(pos_x), int(pos_y)), (int(pos_x1), int(pos_y1)), (0, 0, 255), thickness)

            if len(current_track) > 1:

                (pt1, t1), (pt2, t2) = current_track[0], current_track[1]

                # Get the direction of the vehicle
                direction = get_direction(pt2, pt1)
                object_speed = estimate_speed(pt1, pt2, t1, t2)

                # Adding the speed of the vehicle to the dictionary
                speed_data[track_id].append(object_speed)
            
                # if intersect(current_track[0][0], current_track[1][0], line[0], line[1]):
                #     if direction == "North":
                #         if track_id not in object_couter:
                #             object_couter[track_id] = 1
                #         else:
                #             object_couter[track_id] += 1
                #         count_down += 1
                #     elif direction == "Shout":
                #         count_down += 1
                        



            # -----------------------------

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Write the annotated frame to the output video
        out.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
