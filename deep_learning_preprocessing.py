import cv2
import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import multiprocessing
from functools import partial

def create_metadata(input_dir, output_csv):
    data = []
    for label in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir, label)) or label == "processed_tensors":
            pass
        elif os.path.isdir(os.path.join(input_dir, label)):
            for video_file in [f for f in os.listdir(os.path.join(input_dir, label)) if f.endswith(".mp4")]:
                video_path = os.path.join(input_dir, label, video_file)
                data.append({"video_path": video_path, "label": 1 if label == "manipulated" else 0})

    df = pd.DataFrame(data)
    output_csv = os.path.join(input_dir, output_csv)
    df.to_csv(output_csv, index=False)

    return output_csv

def split_dataset(metadata_csv, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    ratio1 = train_ratio + val_ratio + test_ratio
    ratio2 = val_ratio + test_ratio

    df = pd.read_csv(metadata_csv) 
    train_df, temp_df = train_test_split(df, test_size=val_ratio / ratio1, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_ratio / ratio2, stratify=temp_df['label'], random_state=42)

    train_csv = os.path.join(os.path.dirname(metadata_csv), "train.csv")
    val_csv = os.path.join(os.path.dirname(metadata_csv), "val.csv")
    test_csv = os.path.join(os.path.dirname(metadata_csv), "test.csv")

    print("train", train_df.shape)
    print("val", val_df.shape)
    print("test", test_df.shape)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

def preprocess_video(video_path, frame_count=64, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // frame_count)

    for i in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i*step)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames.append(frame)
        else:
            break

    cap.release()

    while len(frames) < frame_count:
        if frames:
            frames.append(np.zeros_like(frames[-1]))
        else:
            frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    frames = np.array(frames).transpose(3, 0, 1, 2)
    frames = frames.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean = mean[:, None, None, None]
    std = std[:, None, None, None]
    frames = (frames - mean) / std

    return torch.tensor(frames, dtype=torch.float32)

def process_video(row, output_dir):
    """Process a single video and save as a tensor file."""
    video_path = row['video_path']
    tensor_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.pt"
    tensor_path = os.path.join(output_dir, tensor_filename)

    if not os.path.exists(tensor_path):
        try:
            tensor = preprocess_video(video_path)
            torch.save(tensor, tensor_path)
            return f"Processed {video_path}"
        except Exception as e:
            return f"Error processing {video_path}: {str(e)}"
    return f"Already processed {video_path}"

def process_dataset(csv_path, output_dir):
    """Process all videos in the dataset and save as tensor files."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.imap(partial(process_video, output_dir=output_dir), (row for _, row in df.iterrows())), total=len(df), desc="Processing videos"))

    for result in results:
        print(result)

def main(master_path="videos/DFD"):
    processed_dir = "processed_tensors"
    
    # Process each split
    for split in ['train', 'val', 'test']:
    # for split in ['test']:
        csv_path = os.path.join(master_path, f"{split}.csv")
        output_dir = os.path.join(master_path, processed_dir, split)
        process_dataset(csv_path, output_dir)
        print(f"Finished processing {split} split")

if __name__ == "__main__":
    input_dir = "videos/DFD"

    output_csv = create_metadata(input_dir, "metadata.csv")
    split_dataset(output_csv, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    main(input_dir)