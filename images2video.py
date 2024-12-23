# Importing libraries
import os
import cv2
from PIL import Image
import re

# Set path to the Google Drive folder with images
path = "/home/ahmad-maulana/Documents/project/deepfake_detection/preprocessed_frames/test/09__talking_against_wall/debug"
os.chdir(path)

mean_height = 0
mean_width = 0

# Function to extract frame number from filename
def get_frame_number(filename):
    # Extract number from debug_X.png format
    match = re.search(r'debug_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    return -1

# Counting the number of images in the directory
num_of_images = len([file for file in os.listdir('.') if file.endswith((".jpg", ".jpeg", ".png"))])
print("Number of Images:", num_of_images)

# Calculating the mean width and height of all images
for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        im = Image.open(os.path.join(path, file))
        width, height = im.size
        mean_width += width
        mean_height += height

# Averaging width and height
mean_width = int(mean_width / num_of_images)
mean_height = int(mean_height / num_of_images)

# Resizing all images to the mean width and height
for file in os.listdir('.'):
    if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith("png"):
        im = Image.open(os.path.join(path, file))
        # Use Image.LANCZOS instead of Image.ANTIALIAS for downsampling
        im_resized = im.resize((mean_width, mean_height), Image.LANCZOS)
        im_resized.save(file, 'JPEG', quality=95)
        print(f"{file} is resized")


# Function to generate video
def generate_video():
    image_folder = path
    video_name = 'mygeneratedvideo.avi'

    # Get and sort images by frame number
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # Sort numerically by frame number
    images.sort(key=get_frame_number)
    print("\nProcessing frames in order:")
    for img in images[:10]:  # Print first 10 frames to verify order
        print(f"Frame number: {get_frame_number(img)} - {img}")
    if len(images) > 10:
        print("...")

    # Set frame from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file with 24 FPS
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video file
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

# Calling the function to generate the video
generate_video()