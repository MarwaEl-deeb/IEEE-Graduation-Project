import cv2
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, messagebox

window= Tk()
window.geometry("850x650+360+60")
window.title("Body Detection Model")
window.configure(bg = "#73709A")
window.resizable(False, False)

class camera:
    
    def stop(self):
        if self.video_capture.isOpened():
            messagebox.showinfo('Closure', 'Thanks!\nGoodbye')
            self.video_capture.release()
            window.destroy()

    def __init__(self, window):
        self.window = window
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.video_capture = cv2.VideoCapture(0)

        self.current_image = None
        self.cam_frame = tk.Canvas(window, bg='black', width=698,height=480,highlightthickness=0)
        self.cam_frame.place(x=75, y=82)

        self.stop_image = PhotoImage(file="Stop_button.png")
        self.stop_button = Button(
            image=self.stop_image,
            borderwidth=0,
            command=self.stop,
            relief="flat",
            cursor='hand2'
        )
        self.stop_button.place( x=300.0,
                                y=585.0,
                                width=208.0,
                                height=44.0)

        self.text = tk.Label(window,
                                 text=" Your Body Detection ",
                                 font=("times", 33, "italic bold"),
                                 fg="#38355D",
                                 bg='#73709A')
        self.text.place(x=60, y=20)
        self.update_camera()
        self.window.mainloop()

    def update_camera(self):
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            ret, frame = self.video_capture.read()
            if ret:
                self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.results = holistic.process(self.image)
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

                image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                # face detection
                self.mp_drawing.draw_landmarks(self.image, self.results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                          self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                          self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

                # left hand detection
                self.mp_drawing.draw_landmarks(self.image, self.results.left_hand_landmarks,self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(121, 44, 76), thickness=2, circle_radius=2))

                # right hand detection
                self.mp_drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(121, 44, 76), thickness=2, circle_radius=2))

                # pose detection
                self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                          self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                          self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

                rgb_frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                rgb_frame = cv2.resize(rgb_frame, (0, 0), None, 1.1, 1.1)

                self.photo = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
                self.cam_frame.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.current_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)))
                self.window.after(30, self.update_camera)

camera_frame = camera(window)
window.mainloop()

