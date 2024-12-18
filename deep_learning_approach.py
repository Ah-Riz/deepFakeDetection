import cv2
import os
import tqdm
import pandas as pd

def preprocess_video(video_path, output_dir, target_size=(224, 224), fps=24):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, len(video_path.split("/"))):
        output_dir = os.path.join(output_dir, video_path.split("/")[i])
        os.makedirs(output_dir, exist_ok=True)

    for label in os.scandir(video_path):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        for video_file in tqdm.tqdm(os.listdir(os.path.join(video_path, label)), desc=f"Processing {label}"):
            video_path = os.path.join(video_path, label, video_file)
            video = cv2.VideoCapture(video_path)

            cap = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(label_dir, video_file), fourcc, fps, target_size)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                resized_frame = cv2.resize(frame, target_size)
                out.write(resized_frame)

            cap.release()
            out.release()
    return output_dir

def create_metadata(input_dir, output_csv):
    data = []
    for label in os.scandir(input_dir):
        for video_file in os.listdir(os.path.join(input_dir, label)):
            video_path = os.path.join(input_dir, label, video_file)
            data.append({"video_path": video_path, "label": 1 if label.name == "manipulated" else 0})

    df = pd.DataFrame(data)
    output_csv = os.path.join(input_dir, output_csv)
    df.to_csv(output_csv, index=False)

    return output_csv

if __name__ == "__main__":
    input_dir = "videos/DFD"
    output_dir = "DL_preprocessed"

    output_dir = preprocess_video(input_dir, output_dir)
    output_csv = create_metadata(output_dir, "metadata.csv")