import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_official_hmdb import SkeletonDatasetOfficial
from model import ActionRecognitionSTGCN


DATA_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\processed_data_hmdb51"
SPLITS_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\test_train_splits\testTrainMulti_7030_splits"

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
FIXED_FRAMES = 150
NUM_CLASSES = 51

def train_pipeline():
    # Επιλογή Συσκευής (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    # Φόρτωση Δεδομένων με βάση το Official Split 1
    print("\nLoading HMDB51 Official Split 1...")
    train_dataset = SkeletonDatasetOfficial(
        data_dir=DATA_DIR, 
        splits_dir=SPLITS_DIR, 
        split_num=1, 
        is_train=True, 
        fixed_frames=FIXED_FRAMES
    )
    
    val_dataset = SkeletonDatasetOfficial(
        data_dir=DATA_DIR, 
        splits_dir=SPLITS_DIR, 
        split_num=1, 
        is_train=False, 
        fixed_frames=FIXED_FRAMES
    )

    print(f"Training Samples (Official): {len(train_dataset)}")
    print(f"Validation Samples (Official): {len(val_dataset)}")

    # Δημιουργία DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Αρχικοποίηση Μοντέλου, Loss, Optimizer
    model = ActionRecognitionSTGCN(in_channels=8, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0.0
    print("\n--- Starting Official HMDB51 Training Loop ---")
    start_total_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # --- Training Phase ---
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

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # --- Validation Phase ---
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

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Αποθήκευση του καλύτερου μοντέλου για το Official Split
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model_hmdb51_official.pth')
            
        scheduler.step()

    total_duration = (time.time() - start_total_time) / 60
    print(f"\nTraining Complete in {total_duration:.2f} minutes!")
    print(f"Official Validation Accuracy (Split 1): {best_val_acc:.2f}%")
    print("Saved best model weights to 'best_model_hmdb51_official.pth'")

if __name__ == "__main__":
    train_pipeline()