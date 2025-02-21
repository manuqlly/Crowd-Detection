import pandas as pd
import numpy as np
import math

# Load detected persons from CSV
df = pd.read_csv("person_detections.csv")

# Distance threshold for "closeness"
DISTANCE_THRESHOLD = 250  
FRAME_PERSISTENCE = 10  # A group must persist for at least 5 frames

# Function to calculate Euclidean distance
def euclidean_distance(box1, box2):
    x1c, y1c = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2  # Center of person 1
    x2c, y2c = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2  # Center of person 2
    return math.sqrt((x2c - x1c) ** 2 + (y2c - y1c) ** 2)

# Dictionary to track crowds across frames
crowd_tracker = {}

# Store results
crowd_events = []

# Group detections by frame
grouped_frames = df.groupby("Frame")

for frame_num, detections in grouped_frames:
    persons = detections[["Person_ID", "X1", "Y1", "X2", "Y2"]].values  
    print(f"Frame {frame_num}: Found {len(persons)} people")  

    # Find groups of people standing close
    crowd_groups = []
    visited = set()

    for i, person1 in enumerate(persons):
        pid1, *bbox1 = person1  # Extract Person_ID and bounding box

        if pid1 in visited:
            continue

        group = [pid1]
        visited.add(pid1)

        for j, person2 in enumerate(persons):
            pid2, *bbox2 = person2

            if pid2 in visited or pid1 == pid2:
                continue

            if euclidean_distance(bbox1, bbox2) < DISTANCE_THRESHOLD:
                group.append(pid2)
                visited.add(pid2)

        if len(group) >= 3:  # Minimum 3 people = crowd
            crowd_groups.append(group)

    # Convert groups to unique sets
    unique_crowds = {tuple(sorted(group)) for group in crowd_groups}

    # Track persistence
    new_crowd_tracker = {}
    for crowd in unique_crowds:
        if crowd in crowd_tracker:
            new_crowd_tracker[crowd] = crowd_tracker[crowd] + 1
        else:
            new_crowd_tracker[crowd] = 1

    # Log valid crowds (persisted for 5+ frames)
    for key, count in new_crowd_tracker.items():
        if count >= FRAME_PERSISTENCE:
            crowd_events.append((frame_num, len(key), key))  # Store frame, crowd size, and person IDs

    # Update tracker
    crowd_tracker = new_crowd_tracker  

# Save results
# Convert numpy int64 to normal Python int and format Person_IDs properly
df_crowds = pd.DataFrame(crowd_events, columns=["Frame", "Person_Count", "Person_IDs"])
df_crowds["Person_IDs"] = df_crowds["Person_IDs"].apply(lambda x: ", ".join(map(str, x)))  # Clean format

# Save clean CSV
df_crowds.to_csv("crowd_detections.csv", index=False)


if df_crowds.empty:
    print("⚠️ No crowds detected. Try increasing DISTANCE_THRESHOLD.")
else:
    df_crowds.to_csv("crowd_detections.csv", index=False)
    print("✅ Crowd detection complete! Results saved in 'crowd_detections.csv'")
