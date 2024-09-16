import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the provided training data
file_path = 'info.csv'  # Ensure the file is in the same directory or provide the correct path
data = pd.read_csv(file_path)

# Extract the relevant columns
anchors = data.iloc[:, :27]
positives = data.iloc[:, 27:54]
negatives = data.iloc[:, 54:81]

# Normalize the data
anchors = anchors / anchors.max()
positives = positives / positives.max()
negatives = negatives / negatives.max()

# Convert to numpy arrays and then to PyTorch tensors
anchors = torch.tensor(anchors.to_numpy(), dtype=torch.float32)
positives = torch.tensor(positives.to_numpy(), dtype=torch.float32)
negatives = torch.tensor(negatives.to_numpy(), dtype=torch.float32)

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
model = EmbeddingNet(input_dim, embedding_dim)

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

# Initialize loss function and optimizer
triplet_loss = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create DataLoader
train_dataset = TensorDataset(anchors, positives, negatives)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for anchor, positive, negative in train_loader:
        optimizer.zero_grad()
        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
            
    total_loss /= len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'embedding_model.pth')
print("Model saved to embedding_model.pth")
