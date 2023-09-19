import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import math

class Measurement:
    def __init__(self):
        self.points = []
        self.values = []

    def add_point(self, point):
        self.points.append(point)
        if len(self.points) >= 2:
            distance = math.sqrt((self.points[-1][0] - self.points[-2][0])**2 + (self.points[-1][1] - self.points[-2][1])**2)
            width = abs(self.points[-1][0] - self.points[-2][0])
            height = abs(self.points[-1][1] - self.points[-2][1])
            self.values.append((distance, width, height))

    def clear(self):
        self.points.clear()
        self.values.clear()

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Cargar el video
        self.video_source = filedialog.askopenfilename()
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            raise ValueError("No se puede abrir el video", self.video_source)

        self.canvas = tk.Canvas(window, width=1920, height=1080)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.button_frame = tk.Frame(window)
        self.button_frame.grid(row=0, column=1, padx=10, pady=10, sticky="ns")

        self.btn_done = tk.Button(self.button_frame, text="Done", width=10, command=self.save_measurement)
        self.btn_done.pack(pady=10)

        self.btn_clear = tk.Button(self.button_frame, text="Clear", width=10, command=self.clear_measurement)
        self.btn_clear.pack(pady=10)

        self.btn_dummy1 = tk.Button(self.button_frame, text="Botón 3", width=10)
        self.btn_dummy1.pack(pady=10)

        self.btn_dummy2 = tk.Button(self.button_frame, text="Botón 4", width=10)
        self.btn_dummy2.pack(pady=10)

        self.current_measurement = Measurement()

        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.vid.read()
        if not ret:
            return
        resized_image = cv2.resize(frame, (1920, 1080))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags="bg_img")
        self.canvas.tag_lower("bg_img")  # Mueve la imagen al fondo para que no sobrescriba otras marcas
        self.canvas.bind("<Button-1>", self.get_coords)
        self.canvas.bind("<Motion>", self.on_mouse_move)

    def on_mouse_move(self, event):
        self.canvas.delete("cursor_circle")
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="green", width=2, tags="cursor_circle")

    def get_coords(self, event):
        self.current_measurement.add_point((event.x, event.y))
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, fill="red", width=2)

        if len(self.current_measurement.points) >= 2:
            self.canvas.create_line(self.current_measurement.points[-2], self.current_measurement.points[-1], fill="blue", width=2)

    def save_measurement(self):
        # Aquí se puede agregar lógica adicional para procesar/guardar el objeto current_measurement, si es necesario.
        self.current_measurement = Measurement()
        # No borramos el canvas completo, solo las marcas del cursor
        self.canvas.delete("cursor_circle")

    def clear_measurement(self):
        self.current_measurement.clear()
        # Solo se borra el círculo del cursor
        self.canvas.delete("cursor_circle")

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

root = tk.Tk()
root.geometry("2100x1100")
app = App(root, "Video Distancia")
