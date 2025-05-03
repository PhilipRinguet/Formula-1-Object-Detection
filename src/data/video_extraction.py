import os
import cv2
from tqdm import tqdm

def extract_frames(video_path: str, output_dir: str, frame_interval: int = 1) -> None:
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames//frame_interval, desc="Extracting frames") as pbar:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count}min.jpg")
                cv2.imwrite(frame_path, frame)
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()

def process_video_batch(video_dir: str, output_dir: str, frame_interval: int = 1) -> None:
    """
    Process multiple videos in a directory.
    
    Args:
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mkv')):
            video_path = os.path.join(video_dir, video_file)
            video_output_dir = os.path.join(output_dir, os.path.splitext(video_file)[0])
            
            print(f"Processing {video_file}...")
            extract_frames(video_path, video_output_dir, frame_interval)
            print(f"Completed processing {video_file}")