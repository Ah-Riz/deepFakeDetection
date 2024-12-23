import os
import cv2
import mediapipe as mp
import multiprocessing
import pandas as pd
import numpy as np

def compute_ear(eye_points):
    # Compute the eye aspect ratio given eye landmarks
    # The eye landmarks should be in the order: left, right, top, bottom points
    def euclidean_dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Vertical eye landmarks
    v1 = euclidean_dist(eye_points[1], eye_points[5])
    v2 = euclidean_dist(eye_points[2], eye_points[4])
    
    # Horizontal eye landmark
    h = euclidean_dist(eye_points[0], eye_points[3])
    
    # Eye aspect ratio
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
    return ear

def is_blinking(ear, threshold=0.25):
    # Return True if the eye is closed (blinking)
    return ear < threshold

def validate_eye_region(eye_roi, min_brightness=20, min_contrast=15, is_blink=False):
    if eye_roi.size == 0:
        print("Validation failed: Empty eye region")
        return False
        
    # Convert to grayscale for brightness and contrast check
    if len(eye_roi.shape) == 3:
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_eye = eye_roi
        
    # Adjust thresholds if eye is blinking
    actual_min_brightness = min_brightness * 0.7 if is_blink else min_brightness
    actual_min_contrast = min_contrast * 0.7 if is_blink else min_contrast
        
    # Check brightness
    mean_brightness = np.mean(gray_eye)
    if mean_brightness < actual_min_brightness:
        print(f"Validation failed: Brightness {mean_brightness:.2f} < {actual_min_brightness:.2f}")
        return False
        
    # Check contrast
    contrast = np.std(gray_eye)
    if contrast < actual_min_contrast:
        print(f"Validation failed: Contrast {contrast:.2f} < {actual_min_contrast:.2f}")
        return False
        
    return True

def process_single_video(video_info):
    video_path, video_file, output_dir, label = video_info
    
    video_name = video_file.split(".")[0]
    current_output_dir = os.path.join(output_dir, video_name)
    
    # Skip if already processed
    if os.path.exists(current_output_dir) and len(os.listdir(current_output_dir)) > 0:
        print(f"Skipping {video_file} - already processed")
        return None

    # Initialize MediaPipe Face Detection and Face Mesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    try:
        # Initialize video capture
        cap = cv2.VideoCapture(os.path.join(video_path, video_file))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            return None

        # Video properties
        frame_count = 0
        processed_count = 0
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_fps = 24
        frame_interval = int(round(original_fps / target_fps)) if original_fps > target_fps else 1
        video_duration = total_frames / original_fps

        # Metrics initialization
        ear_values = []
        blink_start = None
        blink_durations = []
        total_blinks = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            if frame.shape[0] < 100 or frame.shape[1] < 100:
                frame_count += 1
                continue

            # Process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(rgb_frame)

            if not face_results.detections:
                frame_count += 1
                continue

            # Get face box
            detection = face_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]

            # Calculate face region
            margin = 0.3
            x = max(0, int((bbox.xmin - margin) * w))
            y = max(0, int((bbox.ymin - margin) * h))
            width = min(int((bbox.width + 2 * margin) * w), w - x)
            height = min(int((bbox.height + 2 * margin) * h), h - y)

            box_size = max(width, height)
            x_center = x + width // 2
            y_center = y + height // 2
            x = max(0, x_center - box_size // 2)
            y = max(0, y_center - box_size // 2)
            box_size = min(box_size, w - x, h - y)

            # Process face region
            face_frame = frame[y:y + box_size, x:x + box_size]
            face_frame = cv2.resize(face_frame, (640, 640))
            rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

            # Get face landmarks
            results = face_mesh.process(rgb_face_frame)
            if not results.multi_face_landmarks:
                frame_count += 1
                continue

            face_landmarks = results.multi_face_landmarks[0]

            # Scale landmarks
            scaled_landmarks = []
            for landmark in face_landmarks.landmark:
                scaled_x = int(x + (landmark.x * box_size))
                scaled_y = int(y + (landmark.y * box_size))
                scaled_landmarks.append((scaled_x, scaled_y))

            # Get eye points
            LEFT_EYE = [362, 385, 387, 263, 373, 380]
            RIGHT_EYE = [33, 160, 158, 133, 153, 144]
            LEFT_EYEBROW = [276, 283, 282, 295, 285]
            RIGHT_EYEBROW = [46, 53, 52, 65, 55]

            left_eye_points = [(scaled_landmarks[idx][0], scaled_landmarks[idx][1]) for idx in LEFT_EYE]
            right_eye_points = [(scaled_landmarks[idx][0], scaled_landmarks[idx][1]) for idx in RIGHT_EYE]

            # Calculate EAR
            left_ear = compute_ear(left_eye_points)
            right_ear = compute_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # Check blink state
            left_blink = is_blinking(left_ear)
            right_blink = is_blinking(right_ear)
            is_blink = left_blink and right_blink

            ear_values.append(avg_ear)

            # Track blinks
            if is_blink and blink_start is None:
                blink_start = frame_count / original_fps
            elif not is_blink and blink_start is not None:
                blink_duration = (frame_count / original_fps) - blink_start
                blink_durations.append(blink_duration)
                total_blinks += 1
                blink_start = None

            # Validate eye-eyebrow distance
            left_eyebrow_points = [(scaled_landmarks[idx][0], scaled_landmarks[idx][1]) for idx in LEFT_EYEBROW]
            right_eyebrow_points = [(scaled_landmarks[idx][0], scaled_landmarks[idx][1]) for idx in RIGHT_EYEBROW]

            left_eye_y = sum(p[1] for p in left_eye_points) / len(left_eye_points)
            left_eyebrow_y = sum(p[1] for p in left_eyebrow_points) / len(left_eyebrow_points)
            right_eye_y = sum(p[1] for p in right_eye_points) / len(right_eye_points)
            right_eyebrow_y = sum(p[1] for p in right_eyebrow_points) / len(right_eyebrow_points)

            eye_eyebrow_dist = min(left_eye_y - left_eyebrow_y, right_eye_y - right_eyebrow_y)
            if not (0.02 * h <= eye_eyebrow_dist <= 0.1 * h):
                frame_count += 1
                continue

            # Extract and validate eye regions
            margin_w, margin_h = 0.4, 0.3
            def get_eye_box(points):
                min_x = min(p[0] for p in points)
                min_y = min(p[1] for p in points)
                max_x = max(p[0] for p in points)
                max_y = max(p[1] for p in points)
                width = max_x - min_x
                height = max_y - min_y
                
                min_x = max(0, int(min_x - width * margin_w))
                max_x = min(frame.shape[1], int(max_x + width * margin_w))
                min_y = max(0, int(min_y - height * margin_h))
                max_y = min(frame.shape[0], int(max_y + height * margin_h))
                
                return min_x, min_y, max_x - min_x, max_y - min_y

            left_x, left_y, left_w, left_h = get_eye_box(left_eye_points)
            right_x, right_y, right_w, right_h = get_eye_box(right_eye_points)

            left_eye_roi = frame[left_y:left_y + left_h, left_x:left_x + left_w]
            right_eye_roi = frame[right_y:right_y + right_h, right_x:right_x + right_w]

            if left_eye_roi.size == 0 or right_eye_roi.size == 0:
                frame_count += 1
                continue

            min_height = 5 if (left_blink or right_blink) else 10
            if left_w < 10 or left_h < min_height or right_w < 10 or right_h < min_height:
                frame_count += 1
                continue

            left_valid = validate_eye_region(left_eye_roi, min_brightness=20, min_contrast=15, is_blink=left_blink)
            right_valid = validate_eye_region(right_eye_roi, min_brightness=20, min_contrast=15, is_blink=right_blink)

            if not (left_valid and right_valid):
                frame_count += 1
                continue

            processed_count += 1
            frame_count += 1

        # Calculate metrics
        if ear_values:
            metrics = {
                'video_name': video_file,
                'average_ear': np.mean(ear_values),
                'ear_std_dev': np.std(ear_values),
                'blink_rate': (total_blinks / video_duration) * 60,
                'avg_blink_duration': np.mean(blink_durations) if blink_durations else 0,
                'label': label
            }
            print(f"Processed {video_file} - {processed_count}/{frame_count} frames")
            return metrics

    except Exception as e:
        print(f"Error processing {video_file}: {str(e)}")
        return None

    finally:
        cap.release()
        face_detection.close()
        face_mesh.close()

def process_video_frames(video_path, output_dir, label=0, debug=False):
    # os.chdir('../')
    # Create output directories
    video_list_split = video_path.split("/")
    for i in range(1, len(video_list_split)):
        output_dir = os.path.join(output_dir, video_list_split[i])
        os.makedirs(os.path.join(os.getcwd(), output_dir), exist_ok=True)

    # Get video list
    video_list = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')]
    
    # Prepare video information
    video_info_list = [(video_path, video_file, output_dir, label) for video_file in video_list]
    
    # Initialize multiprocessing pool
    num_processes = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU core free
    print(f"Starting video processing with {num_processes} processes")
    
    # Process videos in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_video, video_info_list)   
    
    # Filter out None results and collect metrics
    video_metrics = [metrics for metrics in results if metrics is not None]
    
    # Save metrics to CSV
    if video_metrics:
        metrics_df = pd.DataFrame(video_metrics)
        csv_path = os.path.join(output_dir, 'video_metrics.csv')
        metrics_df.to_csv(csv_path, index=False)
        print(f"Saved metrics to {csv_path}")
    
    return output_dir
    

if __name__ == "__main__":
    process_video_frames("videos/DFD/original", "eye_blink_data", 0)
    process_video_frames("videos/DFD/manipulated", "eye_blink_data", 1)