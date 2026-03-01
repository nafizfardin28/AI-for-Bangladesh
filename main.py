import os
from preprocess import process_videos, detect_and_crop_faces

# Create required folders
os.makedirs("extracted_frames/real", exist_ok=True)
os.makedirs("extracted_frames/fake", exist_ok=True)
os.makedirs("cropped_faces/real", exist_ok=True)
os.makedirs("cropped_faces/fake", exist_ok=True)


process_videos("dataset/real", "extracted_frames/real", "real")
process_videos("dataset/fake", "extracted_frames/fake", "fake")


detect_and_crop_faces("extracted_frames/real", "cropped_faces/real")
detect_and_crop_faces("extracted_frames/fake", "cropped_faces/fake")

print("Preprocessing Completed!")