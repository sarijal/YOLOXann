import json
import os
import cv2
import pandas as pd
from datetime import timedelta

def create_empty_coco_annotation(output_json="annotation.json"):
    """
    Create an empty COCO annotation file with only the categories defined.
    'images' and 'annotations' remain empty.
    """
    coco_data = {
        "info": {
            "description": "Empty COCO annotation",
            "url": "",
            "version": "1.0",
            "year": 2025,
            "contributor": "",
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "none"},
            {"id": 1, "name": "Person"},
            {"id": 2, "name": "Bicycle"},
            {"id": 3, "name": "Car"},
            {"id": 4, "name": "Motorcycle"},
            {"id": 5, "name": "Bus"},
            {"id": 6, "name": "Truck"},
            {"id": 7, "name": "TrafficLight"},
        ]
    }

    with open(output_json, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Created empty COCO annotation file with categories in '{output_json}'.")

def save_frame(cap, frame_number, output_image_path):
    """
    Extract and save a frame from an open video capture object.
    Returns the saved frame (or None if not successful).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_number} saved as {output_image_path}")
        return frame
    else:
        print(f"Error: Cannot read frame {frame_number}.")
        return None

def tottt(radar_csv, fps_csv, video_path, annotation_json="annotation.json"):
    """
    Match radar timestamps with closest frame timestamps,
    save frames, and update the COCO annotation JSON file with both image info
    and a custom annotation (including radar data) for the car category.
    """
    # Ensure the output folder exists
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)

    # Read the FPS CSV file
    try:
        df_fps = pd.read_csv(fps_csv, parse_dates=['Timestamp'])
    except Exception as e:
        print(f"Error reading {fps_csv}: {e}")
        return

    # Read the Radar CSV file
    try:
        df_radar = pd.read_csv(radar_csv, parse_dates=['Date'])
    except Exception as e:
        print(f"Error reading {radar_csv}: {e}")
        return

    # Filter radar data where TargetIndex is NOT zero
    radar_filtered = df_radar[df_radar["TargetIndex"] != 0]

    # Open the video file once
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Load the annotation JSON once
    try:
        with open(annotation_json, "r") as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error reading {annotation_json}: {e}")
        return

    # Ensure the "images" and "annotations" keys exist
    if "images" not in coco_data:
        coco_data["images"] = []
    if "annotations" not in coco_data:
        coco_data["annotations"] = []

    for _, row in radar_filtered.iterrows():
        input_time = row["Date"]
        target_index = row["TargetIndex"]

        # Compute absolute time differences for all frames
        df_fps["TimeDiff"] = abs(df_fps["Timestamp"] - input_time)

        # Find the row with the smallest time difference
        closest_row = df_fps.loc[df_fps["TimeDiff"].idxmin()]

        # Check if the closest frame is within 500ms of the radar timestamp
        if closest_row["TimeDiff"] <= timedelta(milliseconds=500):
            frame_number = int(closest_row["FrameNumber"])

            # Create an output file name, e.g., "20241124_133806.123_T1.jpg"
            file_name = f"{input_time.strftime('%Y%m%d_%H%M%S.%f')[:-3]}_T{target_index}.jpg"
            output_file_path = os.path.join(output_folder, file_name)

            print(f"Matching frame for {input_time} (TargetIndex {target_index}): {frame_number}")

            # Save the frame using the helper function
            frame = save_frame(cap, frame_number, output_file_path)
            if frame is not None:
                # Get the frame dimensions
                height, width = frame.shape[:2]
                # Create a new unique image id (simply count current images + 1)
                new_image_id = len(coco_data["images"]) + 1

                # Create an image entry following COCO format
                image_entry = {
                    "id": new_image_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": input_time.strftime("%Y-%m-%d %H:%M:%S")
                }
                coco_data["images"].append(image_entry)

                # If a car is confirmed (using your criteria), add a radar annotation for the car
                # Here, we're assuming category_id 3 corresponds to "Car"
                new_annotation_id = len(coco_data["annotations"]) + 1
                annotation_entry = {
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": 3,  # Car category
                    "bbox": [],  # Placeholder; update later when bbox info is available
                    "area": 0,   # This can be calculated later from the bbox
                    "iscrowd": 0,
                    "radar_distance": row.get("Distance", None),
                    "radar_speed": row.get("Speed", None),
                    "radar_snr": row.get("SNR", None)
                }
                coco_data["annotations"].append(annotation_entry)

    # Release the video capture
    cap.release()

    # Write the updated annotation JSON back to disk (only once)
    with open(annotation_json, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"Annotation JSON '{annotation_json}' updated with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations.")




# Step 1: Create the empty annotation file (run once)
create_empty_coco_annotation("annotation.json")

# Step 2: Process the CSV files and video, saving frames and updating JSON annotations
tottt("20241124_134858W_radar.csv", "20241124_134858_fps.csv",
      "20241124_134858_back_camera.mp4", annotation_json="annotation.json")
