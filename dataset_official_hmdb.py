import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDatasetOfficial(Dataset):
    def __init__(self, data_dir, splits_dir, split_num=1, is_train=True, fixed_frames=150):
        self.data_dir = data_dir
        self.fixed_frames = fixed_frames
        self.is_train = is_train
        
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.data_paths = []
        self.labels = []
        
        # Target flag: 1 for Train, 2 for Test/Validation
        target_flag = 1 if is_train else 2
        
        print(f"Parsing official Split {split_num} for {'Train' if is_train else 'Validation'}...")
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, cls_name)
            
            # Βρίσκουμε το αντίστοιχο .txt αρχείο για το split
            split_file = f"{cls_name}_test_split{split_num}.txt"
            split_path = os.path.join(splits_dir, split_file)
            
            if not os.path.exists(split_path):
                print(f"Warning: Split file not found: {split_path}")
                continue
                
            with open(split_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    vid_name = parts[0]
                    flag = int(parts[1])
                    
                    if flag == target_flag:
                        # Αλλάζουμε την κατάληξη από .avi (του txt) σε .npy (τα δεδομένα μας)
                        base_name = os.path.splitext(vid_name)[0]
                        
                        # Ψάχνουμε το αρχείο στο φάκελο
                        npy_filename = None
                        for file in os.listdir(cls_dir):
                            if file.startswith(base_name) and file.endswith('.npy'):
                                npy_filename = file
                                break
                        
                        if npy_filename:
                            self.data_paths.append(os.path.join(cls_dir, npy_filename))
                            self.labels.append(self.class_to_idx[cls_name])
                            
        print(f"Loaded {len(self.data_paths)} samples.")

    def fix_frames_length(self, data):
        current_frames = data.shape[0]
        if current_frames == self.fixed_frames: return data
        elif current_frames > self.fixed_frames: return data[:self.fixed_frames]
        else:
            pad_size = self.fixed_frames - current_frames
            if len(data.shape) == 2:
                padding = np.zeros((pad_size, data.shape[1]), dtype=data.dtype)
            else:
                padding = np.zeros((pad_size, data.shape[1], data.shape[2]), dtype=data.dtype)
            return np.vstack((data, padding))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        file_path = self.data_paths[idx]
        label = self.labels[idx]
        
        data = np.load(file_path)
        data = self.fix_frames_length(data)
        
        if len(data.shape) == 2:
            data = data.reshape(self.fixed_frames, 33, 4)
            
        # Spatial Normalization
        left_hip = data[:, 23, :3]
        right_hip = data[:, 24, :3]
        center = (left_hip + right_hip) / 2.0 
        center = center[:, np.newaxis, :]
        data[:, :, :3] = data[:, :, :3] - center
        
        # Temporal Velocity
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        data = np.concatenate((data, velocity), axis=-1)
        
        # Flatten
        data = data.reshape(self.fixed_frames, -1)
            
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)