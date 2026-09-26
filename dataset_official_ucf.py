import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SkeletonDatasetOfficialUCF(Dataset):
    def __init__(self, data_dir, splits_dir, split_num=1, is_train=True, fixed_frames=150):
        self.data_dir = data_dir
        self.fixed_frames = fixed_frames
        self.is_train = is_train
        
        # Φτιάχνουμε το λεξικό των κλάσεων αλφαβητικά (ίδιο για Train/Test)
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.data_paths = []
        self.labels = []
        
        list_name = f"trainlist0{split_num}.txt" if is_train else f"testlist0{split_num}.txt"
        split_path = os.path.join(splits_dir, list_name)
        
        print(f"Parsing UCF101 Official {'Train' if is_train else 'Validation'} List: {list_name}...")
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο: {split_path}")
            
        with open(split_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Τα αρχεία του UCF101 έχουν τη μορφή "ClassName/v_Action_g01_c01.avi"
                # Στο trainlist έχουν και έναν αριθμό στο τέλος, οπότε κρατάμε μόνο το πρώτο μέρος
                file_path_info = line.split()[0] 
                
                if '/' in file_path_info:
                    cls_name, vid_name = file_path_info.split('/')
                else:
                    continue
                    
                # Αφαιρούμε την κατάληξη .avi
                base_name = os.path.splitext(vid_name)[0]
                
                # Αν η κλάση υπάρχει στα δεδομένα μας (για να αποφύγουμε σφάλματα αν λείπουν αρχεία)
                if cls_name in self.class_to_idx:
                    cls_dir = os.path.join(data_dir, cls_name)
                    
                    # Ψάχνουμε το αντίστοιχο .npy αρχείο
                    npy_filename = f"{base_name}.npy"
                    full_npy_path = os.path.join(cls_dir, npy_filename)
                    
                    # Αν δεν το βρει με το ακριβές όνομα, ψάχνει γενικά στο φάκελο
                    if not os.path.exists(full_npy_path):
                        for file in os.listdir(cls_dir):
                            if file.startswith(base_name) and file.endswith('.npy'):
                                full_npy_path = os.path.join(cls_dir, file)
                                break
                    
                    if os.path.exists(full_npy_path):
                        self.data_paths.append(full_npy_path)
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
            
        left_hip = data[:, 23, :3]
        right_hip = data[:, 24, :3]
        center = (left_hip + right_hip) / 2.0 
        center = center[:, np.newaxis, :]
        data[:, :, :3] = data[:, :, :3] - center
        
        velocity = np.zeros_like(data)
        velocity[1:] = data[1:] - data[:-1]
        data = np.concatenate((data, velocity), axis=-1)
        
        data = data.reshape(self.fixed_frames, -1)
            
        return torch.FloatTensor(data), torch.tensor(label, dtype=torch.long)