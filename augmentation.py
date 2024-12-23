import multiprocessing
from functools import partial
import os
import cv2
import numpy as np
from tqdm import tqdm

def process_single_video(video_file, video_path):
    video = cv2.VideoCapture(os.path.join(video_path, video_file))
    augmentations = []
    
    # Horizontal flip
    if np.random.rand() > 0.2:
        augmentations.append(('flip', None))
    
    # Brightness adjustment
    if np.random.rand() > 0.2:
        brightness_factor = np.random.uniform(0.9, 1.1)
        augmentations.append(('bright', brightness_factor))
    
    # Contrast adjustment
    if np.random.rand() > 0.2:
        contrast_factor = np.random.uniform(0.9, 1.1)
        augmentations.append(('contrast', contrast_factor))
    
    # Gaussian blur
    if np.random.rand() > 0.2:
        blur_factor = np.random.choice([3, 5])  # Kernel sizes for blur
        augmentations.append(('blur', blur_factor))
    
    # Rotation
    if np.random.rand() > 0.2:
        rotate_factor = np.random.choice([-10, 10])
        augmentations.append(('rotate', rotate_factor))
    
    for aug_type, factor in augmentations:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(video_path, f"{video_file.split('.')[0]}_{aug_type}.mp4")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        for _ in tqdm(range(total_frames), desc=f"{aug_type} {video_file}", leave=False):
            ret, frame = video.read()
            if ret:
                if aug_type == 'flip':
                    frame = cv2.flip(frame, 1)
                elif aug_type == 'bright':
                    frame = cv2.convertScaleAbs(frame, alpha=factor, beta=0)
                elif aug_type == 'contrast':
                    frame = cv2.convertScaleAbs(frame, alpha=factor, beta=10)
                elif aug_type == 'blur':
                    frame = cv2.GaussianBlur(frame, (factor, factor), 0)
                elif aug_type == 'rotate':
                    (h, w) = frame.shape[:2]
                    (cX, cY) = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D((cX, cY), factor, 1.0)
                    frame = cv2.warpAffine(frame, M, (w, h))

                out.write(frame)
        out.release()
    
    video.release()
    return f"Processed {video_file}"

def video_augmentation(video_path, reset=True):
    if reset:
        for i in os.listdir(video_path):
            if any(s in i for s in ["flip", "bright", "contrast", "blur", "rotate"]):
                os.remove(os.path.join(video_path, i))
    
    # Get list of video files
    video_files = [f for f in os.listdir(video_path) if f.endswith(".mp4")]
    
    # Create a partial function with fixed video_path
    process_func = partial(process_single_video, video_path=video_path)
    
    # Use number of CPU cores minus 1 to avoid overloading
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    # Create pool and process videos
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, video_files),
            total=len(video_files),
            desc="Processing videos"
        ))
    
    return results

if __name__ == "__main__":
    input_dir = "videos/DFD"

    video_augmentation(os.path.join(input_dir, "original"), True)