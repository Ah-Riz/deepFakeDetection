import os
import cv2
import tqdm

def preprocess_videos(input_dir, output_dir, target_size=(224, 224), fps=24):
    input_dir_list = input_dir.split("/")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, len(input_dir_list)):
        output_dir = os.path.join(output_dir, input_dir_list[i])
        os.makedirs(output_dir, exist_ok=True)
    class_dir = [f.path for f in os.scandir(input_dir) if f.is_dir()]
    for class_dir in class_dir:
        class_name = class_dir.split("/")[-1]
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        video_files = [f.path for f in os.scandir(class_dir) if f.is_file() and f.name.endswith(".mp4")]
    

if __name__ == "__main__":
    input_dir = "videos/DFD/DFD_original_sequences"
    output_dir = "frame_consistency_preprocessed_frames"
    preprocess_videos(input_dir, output_dir)