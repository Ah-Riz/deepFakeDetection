from doctest import master
from turtle import st
from cv2.gapi import video
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchvideo.models import create_resnet
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_video(video_path, frame_count=64, size=(224, 224)): 
    cap = cv2.VideoCapture(video_path)
    frames = []

    # while len(frames) < frame_count:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     # Ensure the frame is in RGB format
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    #     frame = cv2.resize(frame, size)
    #     frames.append(frame)
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

    # Handle case for fewer frames
    while len(frames) < frame_count:
        if frames:
            frames.append(np.zeros_like(frames[-1]))
        else:
            frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    frames = np.array(frames).transpose(3, 0, 1, 2)  # Shape (frame_count, channels, height, width)
    frames = frames.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    mean = mean[:, None, None, None]  # Reshape to (3, 1, 1, 1) for broadcasting
    std = std[:, None, None, None]    # Reshape to (3, 1, 1, 1) for broadcasting
    frames = (frames - mean) / std

    # Normalize with ImageNet stats if needed
    return torch.tensor(frames, dtype=torch.float32).to(device)
    # return frames
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # step = max(1, total_frames // frame_count)

    # for i in range(frame_count):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
    #     ret, frame = cap.read()
    #     if ret:
    #         frame = cv2.resize(frame, size)
    #         frame = frame[:, :, ::-1]  # BGR to RGB
    #         frames.append(frame)
    #     else:
    #         break
    
    # cap.release()

    # while len(frames) < frame_count:
    #     if frames:
    #         frames.append(np.zeros_like(frames[-1]))
    #     else:
    #         frames.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    
    # frames = np.array(frames).transpose(0, 3, 1, 2)# Shape (frame_count, channels, height, width)
    # frames = frames.astype(np.float32) / 255.0

    # # normalize with ImageNet stats
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])
    # frames = (frames - mean[:, None, None]) / std[:, None, None]    

    # frames = np.expand_dims(frames, axis=0)

    # return torch.tensor(frames, dtype=torch.float32)

def load_i3d_model(num_classes=2):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'i3d_r50', pretrained=True)
    model.blocks[-2] = nn.AdaptiveAvgPool3d((1, 1, 1))
    model.blocks[-1] = nn.Sequential(
        nn.Flatten(),
        nn.Linear(1024, num_classes)
    )
    return model.to(device)

def the_trainer(max_epochs=10):
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        # log_every_n_steps=10
        )
    return trainer


class DeepFakeDataset(Dataset):
    def __init__(self, metadata_csv, transforms=None):
        self.data = pd.read_csv(metadata_csv)
        self.transforms = transforms
        self.master_path = os.path.join(*metadata_csv.split("/")[:-1])
        self.processed_dir = os.path.join(self.master_path, "processed_tensors", os.path.basename(metadata_csv).split('.')[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data.iloc[idx]['video_path']
        label = self.data.iloc[idx]['label']
        
        # Load preprocessed tensor
        tensor_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}.pt"
        tensor_path = os.path.join(self.processed_dir, tensor_filename)
        
        if os.path.exists(tensor_path):
            video = torch.load(tensor_path, map_location=device)
        else:
            # Fallback to original preprocessing if tensor doesn't exist
            video = load_and_preprocess_video(video_path)
            
        if self.transforms:
            video = self.transforms(video)
        return video, torch.tensor(label, dtype=torch.long).to(device)

class DeepFakeDetector(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        videos, labels = batch
        videos = videos.to(device)
        labels = labels.to(device)
        output = self(videos)
        loss = self.criterion(output, labels)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        videos, labels = batch
        output = self(videos)
        loss = self.criterion(output, labels)
        acc = (output.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def test_step(self, batch, batch_idx):
        videos, labels = batch
        outputs = self(videos)  # Forward pass
        loss = self.criterion(outputs, labels)  # Compute loss
        preds = outputs.argmax(dim=1)  # Get predicted classes
        acc = (preds == labels).float().mean()  # Calculate accuracy
        
        # Store predictions and labels
        self.test_step_outputs = getattr(self, 'test_step_outputs', [])
        self.test_step_outputs.append({
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return {'loss': loss, 'accuracy': acc}
    
    def on_test_epoch_end(self):
        # Aggregate all predictions and labels
        all_preds = torch.cat([out['preds'] for out in self.test_step_outputs])
        all_labels = torch.cat([out['labels'] for out in self.test_step_outputs])
        
        # Convert to numpy for confusion matrix
        all_preds_np = all_preds.numpy()
        all_labels_np = all_labels.numpy()
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels_np, all_preds_np)
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/DL', exist_ok=True)
        
        # Save predictions and labels
        results_df = pd.DataFrame({
            'true_label': all_labels_np,
            'predicted_label': all_preds_np
        })
        results_df.to_csv('results/DL/test_predictions.csv', index=False)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(cm, 
                           columns=['Predicted Real', 'Predicted Fake'],
                           index=['Actual Real', 'Actual Fake'])
        cm_df.to_csv('results/DL/confusion_matrix.csv')
        
        # Print results
        print("\nTest Results:")
        print(f"Confusion Matrix:\n{cm_df}")
        print("\nClassification Report:")
        print(f"True Positives (Correctly identified fake): {cm[1,1]}")
        print(f"True Negatives (Correctly identified real): {cm[0,0]}")
        print(f"False Positives (Real misclassified as fake): {cm[0,1]}")
        print(f"False Negatives (Fake misclassified as real): {cm[1,0]}")
        
        # Calculate metrics
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Clear the outputs list to free memory
        self.test_step_outputs.clear()

def main(master_path = "videos/DFD"):
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        
    train_csv = os.path.join(master_path, "train.csv")
    val_csv = os.path.join(master_path, "val.csv")
    test_csv = os.path.join(master_path, "test.csv")

    train_dataset = DeepFakeDataset(train_csv)
    val_dataset = DeepFakeDataset(val_csv)
    test_dataset = DeepFakeDataset(test_csv)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    detector = DeepFakeDetector(load_i3d_model())

    detector.to(device)

    trainer = the_trainer(max_epochs=50)
    trainer.fit(detector, train_loader, val_loader)

    model_save_path = "model/deepfake_detector_model.pth"  # Specify the path where you want to save the model
    torch.save(detector.state_dict(), model_save_path)

    trainer.test(detector, test_loader)

    for split, dataset in zip(["train", "val", "test"], [train_dataset, val_dataset, test_dataset]):
        labels = [label for _, label in dataset]
        ori = sum(l == 0 for l in labels)
        fake = sum(l == 1 for l in labels)
        print(f"{split} dataset has {ori} original videos and {fake} fake videos")

def predict_video(video_path, model_save_path = "model/deepfake_detector_model.pth"):
    detector = DeepFakeDetector(load_i3d_model(num_classes=2))
    detector.load_state_dict(torch.load(model_save_path, map_location=device))
    detector.to(device)
    detector.eval()

    video = load_and_preprocess_video(video_path)
    with torch.no_grad():
        output = detector(video)
    prediction = output.argmax(dim=1).item()
    print(f"The video is {'original' if prediction == 0 else 'manipulated'}")

if __name__ == "__main__":
    # print("CUDA Available:", torch.cuda.is_available())
    main()
    import webbrowser
    webbrowser.open("https://www.youtube.com/watch?v=j8SlGbcRYHk&list=OLAK5uy_kjJuAdYj6yawhPjBqCzfAJY7ecabakz0I", new=2)
    # predict_video("videos/DFD/original/01__exit_phone_room.mp4")
    # predict_video("videos/DFD/manipulated/01_02__exit_phone_room__YVGY8LOK.mp4")
    
    # dummy = torch.randn(1, 3, 64, 224, 224)
    # model = load_i3d_model(num_classes=2)
    # output = model(dummy)
    # print(output.shape)