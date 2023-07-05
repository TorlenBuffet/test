import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import os




# Define the deep convolutional network
class LFReconstructionNet(nn.Module):
    def __init__(self):
        super(LFReconstructionNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, 3, padding=1)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# Load the image
img = cv2.imread(r"C:\Users\Owner\Documents\Lab\reserch\Data\test_input01.jpg")
img = cv2.resize(img, dsize=(300, 400))

# Extract EPIs
height, width, channels = img.shape
epi_horizontal = np.zeros((height, width, channels), dtype=np.uint8)
for i in range(width):
    epi_horizontal[:, i, :] = img[:, i, :]
    
# Stack EPIs to reconstruct light field
lf = np.zeros((height, width, width, 3))
for i in range(width):
    lf[:, i, :, :] = epi_horizontal

# Normalize light field
lf = lf / 255.0

# Convert light field to PyTorch tensor
lf_tensor = torch.from_numpy(lf.transpose(0, 3, 1, 2)).float()

# Define the reconstruction network
net = LFReconstructionNet()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
num_epochs = 100
for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    output = net(lf_tensor)
    loss = criterion(output, lf_tensor)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Convert reconstructed light field tensor to numpy array
reconstructed_lf = output.detach().numpy().transpose(0, 2, 3, 1)

# Denormalize the reconstructed light field
reconstructed_lf = np.clip(reconstructed_lf, 0.0, 1.0) * 255.0
reconstructed_lf = reconstructed_lf.astype(np.uint8)

# Display the reconstructed light field
cv2.imshow("Reconstructed Light Field", reconstructed_lf)
cv2.waitKey(0)
