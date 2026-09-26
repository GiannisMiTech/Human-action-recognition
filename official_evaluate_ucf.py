import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from dataset_official_ucf import SkeletonDatasetOfficialUCF
from model import ActionRecognitionSTGCN

DATA_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\processed_data"
SPLITS_DIR = r"E:\μεταπτυχιακό Γιάννης\μαθήματα ΜΠΣ\Διπλωματική\Code\Human_action_recognition_opencv\UCF101TrainTestSplits-RecognitionTask\ucfTrainTestlist"
MODEL_WEIGHTS = 'best_model_ucf101_official_aug.pth'

BATCH_SIZE = 32
FIXED_FRAMES = 150
NUM_CLASSES = 101 

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Φόρτωση του Validation Dataset (Split 1)
    print("Loading Validation Data (Official Split 1) for UCF101...")
    val_dataset = SkeletonDatasetOfficialUCF(
        data_dir=DATA_DIR, 
        splits_dir=SPLITS_DIR, 
        split_num=1, 
        is_train=False, 
        fixed_frames=FIXED_FRAMES
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    classes = val_dataset.classes

    # Φόρτωση του Εκπαιδευμένου Μοντέλου
    print(f"Loading model weights from {MODEL_WEIGHTS}...")
    model = ActionRecognitionSTGCN(in_channels=8, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # Διαδικασία Inference (Προβλέψεις)
    print("Running Inference on Validation Set. Please wait...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Υπολογισμός Μετρικών (Per-Class Accuracy)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Per-Class Metrics - UCF101)")
    print("="*60)
    report = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    print(report)

    # Εύρεση των Top-5 και Bottom-5 δράσεων
    cm = confusion_matrix(all_labels, all_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Αντιμετώπιση NaN (αν κάποια κλάση δεν είχε δείγματα)
    class_accuracies = np.nan_to_num(class_accuracies)
    
    acc_dict = {classes[i]: class_accuracies[i] * 100 for i in range(len(classes))}
    sorted_acc = sorted(acc_dict.items(), key=lambda item: item[1], reverse=True)

    print("\n--- TOP 5 BEST PERFORMING ACTIONS (UCF101) ---")
    for i in range(5):
        print(f"{i+1}. {sorted_acc[i][0]}: {sorted_acc[i][1]:.2f}%")

    print("\n--- TOP 5 WORST PERFORMING ACTIONS (UCF101) ---")
    for i in range(1, 6):
        print(f"{i}. {sorted_acc[-i][0]}: {sorted_acc[-i][1]:.2f}%")

    # Σχεδιασμός και Αποθήκευση Confusion Matrix
    print("\nGenerating Confusion Matrix Plot...")
    plt.figure(figsize=(30, 26)) # έκανα πολύ μεγάλο μέγεθος επειδή είναι 101 κλάσεις
    
    # Χρησιμοποιούμε μικρότερη γραμματοσειρά (annot_kws) αν θέλαμε νούμερα, αλλά τα κρύβουμε (annot=False) για να μη γίνει χαμός
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='d',
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix - UCF101 (Official Split 1)', fontsize=30)
    plt.xlabel('Predicted Label', fontsize=22)
    plt.ylabel('True Label', fontsize=22)
    plt.xticks(rotation=90, fontsize=8)  # Μικρή γραμματοσειρά για να χωρέσουν και οι 101 κλάσεις
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ucf101_official_confusion_matrix_aug.png', dpi=300)
    print("Confusion Matrix saved as 'ucf101_official_confusion_matrix_aug.png'")

if __name__ == "__main__":
    evaluate_model()