import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("models/deepfake_model.keras")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_face(face):
    """Resize and normalize face for model input"""
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def predict_video(video_path):
    """Predict whether a video is Real or Fake"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot open video file!")
        return

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Analyze every 5th frame to speed up processing
        if frame_count % 5 == 0:
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cropped_face = frame[y:y+h, x:x+w]

                if cropped_face.size == 0:
                    continue

                processed_face = preprocess_face(cropped_face)
                pred = model.predict(processed_face, verbose=0)[0][0]
                predictions.append(pred)

        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        messagebox.showwarning("Result", "No faces detected in video!")
        return

    avg_score = np.mean(predictions)  

    if avg_score > 0.5:  
        result = f"FAKE VIDEO ({confidence}% confidence)"
    else:
        confidence = round((1 - avg_score) * 100, 2)
        result = f"REAL VIDEO ({confidence}% confidence)"

    messagebox.showinfo("Prediction Result", result)


def browse_file():
    """Open file dialog to select a video and predict"""
    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
    )
    if video_path:
        predict_video(video_path)

root = tk.Tk()
root.title("Deepfake Video Detector")
root.geometry("400x200")

tk.Label(root, text="Deepfake Video Detector", font=("Helvetica", 16)).pack(pady=20)
tk.Button(root, text="Select Video", command=browse_file, width=25, height=2).pack(pady=10)
tk.Button(root, text="Exit", command=root.quit, width=25, height=2).pack(pady=10)

root.mainloop()