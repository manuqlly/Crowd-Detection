import cv2
import os
import pandas as pd
import torch
from ultralytics import YOLO
import sys
sys.path.append("D:/guthib/Resolute AI Software Private Limited/Crowd Detection/sort")
from sort import Sort


# Paths
video_path = "dataset_video.mp4"
output_csv = "person_detections.csv"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  

# Initialize SORT Tracker
tracker = Sort()

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0
detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)

    # Collect YOLO detections
    yolo_detections = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # Class 0 = "person"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])  # Confidence score
                yolo_detections.append([x1, y1, x2, y2, conf])  

    # Convert to tensor & track
    if yolo_detections:
        yolo_detections = torch.tensor(yolo_detections)
        tracked_objects = tracker.update(yolo_detections)

        # Store results (Frame, Person_ID, BBox)
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            detections.append((frame_count, track_id, x1, y1, x2, y2))

cap.release()

# Save results to CSV
df = pd.DataFrame(detections, columns=["Frame", "Person_ID", "X1", "Y1", "X2", "Y2"])
df.to_csv(output_csv, index=False)
print(f"✅ Detections saved to {output_csv}")
