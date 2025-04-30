import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Mount Google Drive automatically
if not os.path.exists('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')

# Load config
config_path = '/content/drive/MyDrive/Colab Notebooks/NFFA-SEM/configs/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# # Load config
# with open('configs/config.yaml', 'r') as f:
#     config = yaml.safe_load(f)

base_dir = config['base_dir']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']
input_size = tuple(config['input_size'])
random_seed = config['random_seed']
subset_ratio = config['subset_ratio']
save_model_path = config['save_model_path']
logs_dir = config['logs_dir']

# Set seed for reproducibility
torch.manual_seed(random_seed)

# Transforms
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
])

# Load full dataset
full_dataset = datasets.ImageFolder(root=base_dir, transform=transform)

# Reduce dataset size
subset_size = int(len(full_dataset) * subset_ratio)
full_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size], generator=torch.Generator().manual_seed(random_seed))

# Split dataset (8:1:1)
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size
print(train_size,val_size,test_size)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.dataset.classes))  # replace last layer

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training history
train_losses = []
train_accuracies = []
val_accuracies = []

# Make sure logs directory exists
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct.double() / len(train_loader.dataset)

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_acc = val_correct.double() / len(val_loader.dataset)
    val_accuracies.append(val_acc.item())

    print(f"Validation Acc: {val_acc:.4f}")

# Final Testing
model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

test_acc = test_correct.double() / len(test_loader.dataset)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
torch.save(model.state_dict(), save_model_path)
print("Model saved!")

# Plot Training Loss
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.savefig(os.path.join(logs_dir, 'training_loss_curve.png'))
plt.show()

# Plot Training and Validation Accuracy
plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curve')
plt.legend()
plt.grid()
plt.savefig(os.path.join(logs_dir, 'training_accuracy_curve.png'))
plt.show()
