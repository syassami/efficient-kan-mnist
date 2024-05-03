import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from src.efficient_kan import KAN

# Load MNIST dataset
train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor()
)

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

# Define the KAN model
kan = KAN([784, 10], base_activation=nn.Identity, grid_size=10)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(kan.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    kan.train()
    train_loss = 0.0
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, (data, target) in enumerate(pbar):
            data = data.view(data.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            output = kan(data)
            loss = criterion(output, target)
            reg_loss = kan.regularization_loss(1, 0)
            total_loss = loss + 1e-5 * reg_loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
            pbar.set_postfix(loss=total_loss.item())

    # Evaluate on the test set
    kan.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)  # Flatten the images
            output = kan(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%"
    )

# Print the learned spline weights
for layer in kan.layers:
    print(layer.spline_weight)
