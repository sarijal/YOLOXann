import json
import os
import cv2
import pandas as pd
from datetime import timedelta
import argparse

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
    # dataset_name = os.path.splitext(os.path.basename(fps_csv_path))[0].replace("_fps", "")
    dataset_name = "mycvatout"
    dataset_root = os.path.join("datasets", dataset_name)
    
    os.makedirs(dataset_root, exist_ok=True)
    annotations_dir = os.path.join(dataset_root, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    images_dir = os.path.join(dataset_root, "images", "train")
    os.makedirs(images_dir, exist_ok=True)
    
    annotation_file = os.path.join(annotations_dir, "instances_train.json")
    
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
        
    for _, radar_row in radar_filtered.iterrows():
        timestamp = radar_row["Date"]
        target_index = radar_row["TargetIndex"]
        tdistance=radar_row["Distance"]
        tspeed=radar_row["Speed"]
        tnoise=radar_row["SNR"]
        
        df_fps["TimeDiff"] = abs(df_fps["Timestamp"] - timestamp)
        closest_frame = df_fps.loc[df_fps["TimeDiff"].idxmin()]
        
        if closest_frame["TimeDiff"] <= timedelta(milliseconds=500):
            frame_idx = int(closest_frame["FrameNumber"])
            file_name = f"F{frame_idx}_D{tdistance}_S{tspeed}_N{tnoise}_T{target_index}.jpg"
            output_path = os.path.join(images_dir, file_name)
            
            frame = save_video_frame(cap, frame_idx, output_path)

    cap.release()
    
# generate_dataset("datasets/20241124_133806W_radar.csv", "datasets/20241124_133806_fps.csv", "datasets/20241124_133806_back_camera.mp4")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from radar, fps CSV files and video.")
    parser.add_argument("radar_csv", type=str, help="Path to the radar CSV file")
    parser.add_argument("fps_csv", type=str, help="Path to the FPS CSV file")
    parser.add_argument("video", type=str, help="Path to the video file")

    args = parser.parse_args()

    generate_dataset(args.radar_csv, args.fps_csv, args.video)