import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Load the provided training data
file_path = 'info.csv'  # Ensure the file is in the same directory or provide the correct path
data = pd.read_csv(file_path)

# Extract the relevant columns
anchors = data.iloc[:, :27]
positives = data.iloc[:, 27:54]
negatives = data.iloc[:, 54:81]
world_states = data.iloc[:, 81:]

# Check for NaNs in the data
print(f"Any NaNs in anchors: {anchors.isna().any().any()}")
print(f"Any NaNs in positives: {positives.isna().any().any()}")
print(f"Any NaNs in negatives: {negatives.isna().any().any()}")
print(f"Any NaNs in world_states: {world_states.isna().any().any()}")

# Normalize the data
anchors = anchors / anchors.max()
positives = positives / positives.max()
negatives = negatives / negatives.max()
world_states = world_states / world_states.max()

# Replace any potential NaNs introduced during normalization with 0
anchors = anchors.fillna(0)
positives = positives.fillna(0)
negatives = negatives.fillna(0)
world_states = world_states.fillna(0)

# Convert to numpy arrays and then to PyTorch tensors
anchors = torch.tensor(anchors.to_numpy(), dtype=torch.float32)
positives = torch.tensor(positives.to_numpy(), dtype=torch.float32)
negatives = torch.tensor(negatives.to_numpy(), dtype=torch.float32)
world_states = torch.tensor(world_states.to_numpy(), dtype=torch.float32)

# Define the model
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(EmbeddingNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
            
    def forward(self, x):
        return self.model(x)

input_dim = 27
embedding_dim = 32
model1 = EmbeddingNet(input_dim, embedding_dim)

# Define the Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
            
    def forward(self, anchor, positive, negative):
        positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
        loss = torch.mean(torch.relu(positive_distance - negative_distance + self.margin))
        return loss

# Hyperparameters
learning_rate = 0.001
num_epochs = 50
batch_size = 32

# Initialize loss function and optimizer for the first model
triplet_loss = TripletLoss(margin=1.0)
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)

# Create DataLoader for the first model
train_dataset1 = TensorDataset(anchors, positives, negatives)
train_loader1 = DataLoader(train_dataset1, batch_size=batch_size, shuffle=True)

# Training loop for the first model
for epoch in range(num_epochs):
    model1.train()
    total_loss1 = 0
    for anchor, positive, negative in train_loader1:
        optimizer1.zero_grad()
        anchor_embedding = model1(anchor)
        positive_embedding = model1(positive)
        negative_embedding = model1(negative)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer1.step()
        total_loss1 += loss.item()
            
    total_loss1 /= len(train_loader1)
    print(f'Embedding Model Epoch {epoch+1}/{num_epochs}, Loss: {total_loss1:.4f}')

# Define the second model for reconstructing the world state
class ClusterToWorld(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClusterToWorld, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.model(x)

world_output_dim = world_states.shape[1]
model2 = ClusterToWorld(embedding_dim, world_output_dim)

# Define loss function and optimizer for the second model
criterion = nn.MSELoss()
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

# Prepare data for the second model
embeddings = model1(anchors).detach()  # Use embeddings from the first model
train_dataset2 = TensorDataset(embeddings, world_states)
train_loader2 = DataLoader(train_dataset2, batch_size=batch_size, shuffle=True)

# Training loop for the second model
for epoch in range(num_epochs):
    model2.train()
    total_loss2 = 0
    for embedding, world_state in train_loader2:
        optimizer2.zero_grad()
        output = model2(embedding)

        # Check for NaNs in the output and world_state
        if torch.isnan(output).any():
            print(f"NaN detected in output at epoch {epoch+1}")
        if torch.isnan(world_state).any():
            print(f"NaN detected in world_state at epoch {epoch+1}")

        loss = criterion(output, world_state)
        if torch.isnan(loss):
            print(f"NaN loss detected at epoch {epoch+1}")
            break
        loss.backward()
        optimizer2.step()
        total_loss2 += loss.item()
            
    total_loss2 /= len(train_loader2)
    if torch.isnan(torch.tensor(total_loss2)):
        break
    print(f'Cluster to World Model Epoch {epoch+1}/{num_epochs}, Loss: {total_loss2:.4f}')

# Create the "models" subfolder if it doesn't exist
folder = 'models'
os.makedirs(folder, exist_ok=True)

# Save both models with suffixes that include the respective losses
filename_base = os.path.splitext(os.path.basename(__file__))[0]
model1_filename = f'{folder}/{filename_base}_embedding_model_{total_loss1:.4f}.pth'
model2_filename = f'{folder}/{filename_base}_cluster_to_world_model_{total_loss2:.4f}.pth'

torch.save(model1.state_dict(), model1_filename)
torch.save(model2.state_dict(), model2_filename)

print(f'Models saved to {model1_filename} and {model2_filename}')

