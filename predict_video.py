import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224


model = load_model("models/deepfake_model.keras")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_count % 5 == 0:

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cropped = frame[y:y+h, x:x+w]

                if cropped.size == 0:
                    continue

                cropped = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                cropped = cropped / 255.0
                cropped = np.expand_dims(cropped, axis=0)

                prediction = model.predict(cropped, verbose=0)[0][0]
                print("Frame prediction:", round(float(prediction), 4))
                predictions.append(prediction)

        frame_count += 1

    cap.release()

    if len(predictions) == 0:
        print("\nNo faces detected in video.")
        return

    high_conf = [p for p in predictions if p > 0.6]
    avg_score = np.mean(high_conf) if len(high_conf) > 0 else np.mean(predictions)

    print("\nTotal Faces Analyzed:", len(predictions))
    print("Average Fake Probability:", round(avg_score, 4))

    if avg_score > 0.6:
        confidence = round(avg_score * 100, 2)
        print(f"\nFINAL RESULT: FAKE VIDEO ({confidence}% confidence)")
    else:
        confidence = round((1 - avg_score) * 100, 2)
        print(f"\nFINAL RESULT: REAL VIDEO ({confidence}% confidence)")


if __name__ == "__main__":
    video_path = input("Enter video path: ")
    predict_video(video_path)