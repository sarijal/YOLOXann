import json
import os
import cv2
import pandas as pd
from datetime import timedelta

def create_empty_coco_annotation(output_json="annotation.json"):
    """
    Creates an empty COCO annotation file with predefined categories.
    """
    coco_structure = {
        "info": {
            "description": "Empty COCO annotation",
            "version": "1.0",
            "year": 2025,
            "date_created": ""
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "person"},
            {"id": 1, "name": "bicycle"},
            {"id": 2, "name": "Car"},
            {"id": 3, "name": "motorcycle"},
            {"id": 4, "name": "airplane"},
            {"id": 5, "name": "bus"},
            {"id": 6, "name": "train"},
            {"id": 7, "name": "truck"}
        ]
    }
    
    try:
        with open(output_json, "w") as f:
            json.dump(coco_structure, f, indent=2)
        print(f"Created empty COCO annotation file: {output_json}")
    except Exception as e:
        print(f"Error writing {output_json}: {e}")

def save_video_frame(video_capture, frame_index, output_path):
    """
    Extracts and saves a specific frame from a video.
    """
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = video_capture.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_index} as {output_path}")
        return frame
    else:
        print(f"Error: Unable to read frame {frame_index}.")
        return None

def generate_dataset(radar_csv_path, fps_csv_path, video_path):
    """
    Creates dataset directories, extracts frames, and updates COCO annotations.
    """
    dataset_name = os.path.splitext(os.path.basename(fps_csv_path))[0].replace("_fps", "")
    dataset_root = os.path.join("datasets", dataset_name)
    
    os.makedirs(dataset_root, exist_ok=True)
    annotations_dir = os.path.join(dataset_root, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    images_dir = os.path.join(dataset_root, "images", "train")
    os.makedirs(images_dir, exist_ok=True)
    
    annotation_file = os.path.join(annotations_dir, "instances_train.json")
    create_empty_coco_annotation(annotation_file)
    
    try:
        df_fps = pd.read_csv(fps_csv_path, parse_dates=['Timestamp'])
        df_radar = pd.read_csv(radar_csv_path, parse_dates=['Date'])
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    radar_filtered = df_radar[df_radar["TargetIndex"] != 0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    try:
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)
    except Exception as e:
        print(f"Error reading {annotation_file}: {e}")
        return
    
    for _, radar_row in radar_filtered.iterrows():
        timestamp = radar_row["Date"]
        target_index = radar_row["TargetIndex"]
        
        df_fps["TimeDiff"] = abs(df_fps["Timestamp"] - timestamp)
        closest_frame = df_fps.loc[df_fps["TimeDiff"].idxmin()]
        
        if closest_frame["TimeDiff"] <= timedelta(milliseconds=500):
            frame_idx = int(closest_frame["FrameNumber"])
            file_name = f"{timestamp.strftime('%Y%m%d_%H%M%S.%f')[:-3]}_T{target_index}.jpg"
            output_path = os.path.join(images_dir, file_name)
            
            frame = save_video_frame(cap, frame_idx, output_path)
            if frame is not None:
                image_id = len(coco_data["images"]) + 1
                height, width = frame.shape[:2]
                
                image_entry = {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name,
                    "date_captured": timestamp.strftime("%Y-%m-%d %H:%M:%S")
                }
                coco_data["images"].append(image_entry)
                
                annotation_id = len(coco_data["annotations"]) + 1
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 2,  # Car category
                    "bbox": [],  # Placeholder; update later when bbox info is available
                    "area": 0,
                    "iscrowd": 0,
                    "radar_distance": radar_row.get("Distance"),
                    "radar_speed": radar_row.get("Speed"),
                    "radar_snr": radar_row.get("SNR")
                }
                coco_data["annotations"].append(annotation_entry)
    
    cap.release()
    
    try:
        with open(annotation_file, "w") as f:
            json.dump(coco_data, f, indent=2)
        print(f"Updated annotation JSON: {annotation_file}")
    except Exception as e:
        print(f"Error writing {annotation_file}: {e}")

generate_dataset("datasets/20241124_133806W_radar.csv", "datasets/20241124_133806_fps.csv", "datasets/20241124_133806_back_camera.mp4")
