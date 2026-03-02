import cv2
import os
from tqdm import tqdm
from mtcnn import MTCNN

def extract_frames(video_path, output_folder, label):
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % 2 == 0:
            filename = os.path.join(output_folder, f"{label}_{frame_id}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1

        count += 1

    cap.release()


def process_videos(input_folder, output_folder, label):
    for video in tqdm(os.listdir(input_folder)):
        video_path = os.path.join(input_folder, video)
        extract_frames(video_path, output_folder, label)

detector = MTCNN()

def detect_and_crop_faces(input_folder, output_folder):
    for img_name in tqdm(os.listdir(input_folder)):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        results = detector.detect_faces(image)

        for result in results:
            x, y, w, h = result['box']
            face = image[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (224, 224))
            cv2.imwrite(os.path.join(output_folder, img_name), face)