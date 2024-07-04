import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = []
        with open(csv_file, newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                row = [int(item) for item in row]
                features = row[:64]
                labels = row[64:]
                self.data.append((features, labels))
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, labels = self.data[idx]
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return features, labels

dataset = CustomDataset('arrays_output.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define neural network model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20

for epoch in range(num_epochs):
    for features, labels in dataloader:
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
                
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete!")
