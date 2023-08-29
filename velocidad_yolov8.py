from collections import defaultdict

import cv2
import numpy as np
import time

from collections import deque
from functions import *

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')
# https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
classes_dict = {
    0: 'person',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    }
classes_list=list(classes_dict.keys())


# Open the video file
video_path = "./velocidad_pic/7.mp4"


cap = cv2.VideoCapture(video_path)
# Cambiamos la resolucion de la camara a 1920x1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

# Create a video writer for the output video
output_path = "./runs/1_annotated.mp4"
codec = cv2.VideoWriter_fourcc(*"XVID")  # You can change the codec as needed
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_path, codec, fps, frame_size)



count_up = 0
count_down = 0

# Store the track history
track_history = defaultdict(lambda: [])
speed_data = defaultdict(lambda: [])
counter_enter = defaultdict(lambda: [])
counter_exit = defaultdict(lambda: [])



# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    print(frame.shape)

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame, 
            persist=True,
            conf=0.7,
            iou=0.7,
            device=0,
            tracker='botsort.yaml',
            classes=classes_list,)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh
        

        try:
            track_ids = results[0].boxes.id.int().tolist()
        except:
            track_ids = []

        # Visualize the results on the frame
        annotated_frame = results[0].plot(
            line_width=2,  # thickness of bounding box and track line
            conf=False,  # show confidence score
            font_size=2,  # font size of track ID
        )
        # Draw the line in the middle of the frame
        line = [(0, int(annotated_frame.shape[0] / 2)), (annotated_frame.shape[1], int(annotated_frame.shape[0] / 2))]
        cv2.line(annotated_frame, line[0], line[1], (46, 162, 112), 3)


        # cv2.line(annotated_frame, (0, int(737.94654122)), (annotated_frame.shape[1], int(-0.22085729 * annotated_frame.shape[1] + 737.94654122)), (0, 0, 255), 3)

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):

            box_xc, box_yc, box_w, box_h = box


            # --------Tracking--------
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=10)
                speed_data[track_id] = deque(maxlen=10)

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
                speed_data[track_id].appendleft(object_speed)
                avg_speed = int(sum(speed_data[track_id]) / len(speed_data[track_id]))
                cv2.putText(annotated_frame, f'{avg_speed} km/h', (int(pt1[0] - 15), int(pt1[1] - 15)), 0, 0.75, [0, 0, 255], thickness=2, lineType=cv2.LINE_AA)
                

            
                if intersect(line[0], line[1],current_track[0][0], current_track[1][0]):
                    if direction == "South":
                        if track_id not in counter_enter:
                            counter_enter[track_id] = 1
                        else:
                            counter_enter[track_id] += 1
                        count_up += 1
                    elif direction == "North":
                        if track_id not in counter_exit:
                            counter_exit[track_id] = 1
                        else:
                            counter_exit[track_id] += 1
                        count_down += 1
            # -----------------------------
                


        for idx, (key, value) in enumerate(counter_enter.items()): # Counter Enter
            counter_enter_str = str(key) + ":" + str(value)
            # Cars Entering
            cv2.putText(annotated_frame, f'Vehicles Entering', (frame.shape[1] - 500, 35), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            ###
            cv2.putText(annotated_frame, "Total Count: "+str(count_up), (frame.shape[1] - 500, 75), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)

            ###
            cv2.putText(annotated_frame, counter_enter_str, (frame.shape[1] - 200, 105 + (idx*40)), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)

        for idx, (key, value) in enumerate(counter_exit.items()): # Counter Exit
            counter_exit_str = str(key) + ":" + str(value)
            # Cars Leaving
            cv2.putText(annotated_frame, f'Vehicles Leaving', (11, 35), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)
            ###
            cv2.putText(annotated_frame, "Total Count: " +str(count_down), (11, 75), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)

            ###
            cv2.putText(annotated_frame, counter_exit_str, (11, 105+ (idx*40)), 0, 1, [255, 0, 0], thickness=2, lineType=cv2.LINE_AA)


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
