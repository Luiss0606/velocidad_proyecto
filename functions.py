import math
import pickle
import cv2

from sklearn.preprocessing import PolynomialFeatures



def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def get_direction(last_pos, first_pos):
    direction = ""

    if first_pos[1] < last_pos[1]:
        direction += "South"
    elif first_pos[1] > last_pos[1]:
        direction += "North"
    else:
        direction += ""
    return direction


def ppm_calculate(x_pos, y_pos):
    with open('./velocidad_pic/ppm.pkl', 'rb') as f:
        model = pickle.load(f)
    return model.predict(PolynomialFeatures(2).fit_transform([[x_pos, y_pos]]))[0]


def estimate_speed(first_pos, last_pos, first_t, last_t):
    d_pixels = math.sqrt(math.pow(last_pos[0] - first_pos[0], 2) + math.pow(last_pos[1] - first_pos[1], 2))
    middle_point = (int((last_pos[0] + first_pos[0]) / 2), int((last_pos[1] + first_pos[1]) / 2))
    ppm = ppm_calculate(middle_point[0], middle_point[1])
    d_meters = d_pixels / ppm
    # Taking the time in seconds, we take te positive value of the difference
    dt = abs(last_t - first_t)
    speed = d_meters / dt  # in m/s
    speed = speed * 3.6  # convert to km/h
    print("Speed: ", speed)
    print("Pixels: ", d_pixels)
    print("PPM: ", ppm)
    print("Meters: ", d_meters)
    print("Time: ", dt)
    return int(speed)

# Save the output video
def save_output(video_capture):
    output_path = "./runs/video_output.mp4"
    codec = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                  int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, codec, fps, frame_size)
    return out