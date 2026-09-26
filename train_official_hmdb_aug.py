import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_official_hmdb_aug import SkeletonDatasetOfficialHMDBAug
from model import ActionRecognitionSTGCN

DATA_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\processed_data_hmdb51"
SPLITS_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\test_train_splits\testTrainMulti_7030_splits"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
FIXED_FRAMES = 150
NUM_CLASSES = 51

def train_pipeline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading HMDB51 Official Split 1 with ARTIFICIAL OCCLUSION...")
    train_dataset = SkeletonDatasetOfficialHMDBAug(
        data_dir=DATA_DIR, 
        splits_dir=SPLITS_DIR, 
        split_num=1, 
        is_train=True, 
        fixed_frames=FIXED_FRAMES
    )
    
    val_dataset = SkeletonDatasetOfficialHMDBAug(
        data_dir=DATA_DIR, 
        splits_dir=SPLITS_DIR, 
        split_num=1, 
        is_train=False, 
        fixed_frames=FIXED_FRAMES
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ActionRecognitionSTGCN(in_channels=8, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    print("\n--- Starting Official HMDB51 Training Loop (AUGMENTED) ---")
    start_total_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | "
              f"Train Acc: {(train_correct / train_total) * 100:.2f}% | "
              f"Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model_hmdb51_official_aug.pth')
            
        scheduler.step()

    print(f"\nTraining Complete in {(time.time() - start_total_time) / 60:.2f} minutes!")
    print(f"Official Validation Accuracy (HMDB51 Occlusion): {best_val_acc:.2f}%")
    print("Saved best model to 'best_model_hmdb51_official_aug.pth'")

if __name__ == "__main__":
    train_pipeline()