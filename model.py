import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MediaPipeGraph:
    def __init__(self):
        self.num_nodes = 33
        self.edges = [
            (0,1), (1,2), (2,3), (3,7),
            (0,4), (4,5), (5,6), (6,8),
            (9,10),
            (11,12), (11,13), (13,15), (15,17), (15,19), (15,21), (17,19),
            (12,14), (14,16), (16,18), (16,20), (16,22), (18,20),
            (11,23), (12,24), (23,24),
            (23,25), (25,27), (27,29), (29,31), (27,31),
            (24,26), (26,28), (28,30), (30,32), (28,32)
        ]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        # Create an empty 33x33 matrix
        A = np.zeros((self.num_nodes, self.num_nodes))
        # Draw the connections
        for i, j in self.edges:
            A[i, j] = 1
            A[j, i] = 1
        # Add self-loops (a joint is connected to itself)
        A = A + np.eye(self.num_nodes)
        
        # Normalize the matrix 
        D = np.diag(np.sum(A, axis=1) ** -0.5)
        A = np.dot(np.dot(D, A), D)
        return torch.tensor(A, dtype=torch.float32)

class ST_GCN_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(ST_GCN_Layer, self).__init__()
        self.register_buffer('A', A) 
        
        # Analyzes the skeleton posture
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Analyzes how the posture changes over time
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), 
                                       padding=(4, 0), stride=(stride, 1))
        
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.spatial_conv(x)
        x = torch.matmul(x, self.A) 

        x = self.temporal_conv(x)
        x = self.batch_norm(x)
        return self.relu(x)

class ActionRecognitionSTGCN(nn.Module):
    def __init__(self, in_channels=8, num_classes=101):
        super(ActionRecognitionSTGCN, self).__init__()
        
        # Build the skeleton graph
        graph = MediaPipeGraph()
        self.A = graph.A
        

        self.st_gcn1 = ST_GCN_Layer(in_channels, 64, self.A)
        self.st_gcn2 = ST_GCN_Layer(64, 128, self.A, stride=2)
        self.st_gcn3 = ST_GCN_Layer(128, 256, self.A, stride=2)

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        Batch, Frames, _ = x.size()
        Channels = 8  # (X, Y, Z, Vis, dX, dY, dZ, dVis)
        Nodes = 33    # 33 MediaPipe Joints
        
        # Reshape data into the required 4D tensor for ST-GCN: (Batch, Channels, Frames, Nodes)
        x = x.view(Batch, Frames, Nodes, Channels)
        x = x.permute(0, 3, 1, 2).contiguous() 
        
        # Pass through ST-GCN layers
        x = self.st_gcn1(x)
        x = self.st_gcn2(x)
        x = self.st_gcn3(x)
        
        # Global Average Pooling 
        x = F.avg_pool2d(x, x.size()[2:]) 
        x = x.view(Batch, -1) 
        
        # Output probabilities for the 101 classes
        out = self.fc(x)
        return out

# Diagnostic Test Block
if __name__ == "__main__":
    print("Initializing ST-GCN Model...")
    dummy_input = torch.randn(4, 150, 264) # Batch of 4, 150 frames, 264 flattened features
    model = ActionRecognitionSTGCN(num_classes=101)
    output = model(dummy_input)
    
    print("\n--- Diagnostic Results ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape} -> (Expected: [4, 101])")
    print("Network successfully mapped the MediaPipe skeleton!")