import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# === 1. 引数処理 ===
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--batch_size', type=int, default=None)
args = parser.parse_args()

# === 2. 設定読み込み & 上書き対応 ===
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

if args.lr is not None:
    config['learning_rate'] = args.lr
if args.batch_size is not None:
    config['batch_size'] = args.batch_size

# === 3. 設定変数展開 ===
base_dir = config['base_dir']
batch_size = config['batch_size']
num_epochs = config['num_epochs']
learning_rate = config['learning_rate']
input_size = tuple(config['input_size'])
random_seed = config['random_seed']
subset_ratio = config['subset_ratio']
output_base = config.get('output_base_dir', 'outputs')

# === 4. 出力ディレクトリ ===
run_name = datetime.now().strftime('%Y%m%d_%H%M%S') + f"_lr{learning_rate}_bs{batch_size}"
#output_dir = os.path.join("outputs", run_name)
output_dir = os.path.join(output_base, run_name)
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, 'train.log')
model_path = os.path.join(output_dir, 'model.pth')
loss_plot_path = os.path.join(output_dir, 'training_loss_curve.png')
acc_plot_path = os.path.join(output_dir, 'training_accuracy_curve.png')
writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

def log(msg):
    print(msg)
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
log(f"Training started: {run_name}")

# === 5. シード固定 ===
torch.manual_seed(random_seed)

# === 6. データ準備 ===
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=base_dir, transform=transform)
subset_size = int(len(full_dataset) * subset_ratio)
full_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size],
                               generator=torch.Generator().manual_seed(random_seed))

train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

log(f"Dataset sizes: train={train_size}, val={val_size}, test={test_size}")

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# === 7. モデルと学習準備 ===
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(full_dataset.dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, train_accuracies, val_accuracies = [], [], []

# === 8. 学習ループ ===
for epoch in range(num_epochs):
    model.train()
    running_loss, correct = 0.0, 0

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

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)
    log(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

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
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    log(f"[Epoch {epoch+1}] Validation Acc: {val_acc:.4f}")

# === 9. テスト評価 ===
model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

test_acc = test_correct.double() / len(test_loader.dataset)
log(f"Test Accuracy: {test_acc:.4f}")
writer.add_hparams({'lr': learning_rate, 'bs': batch_size}, {'test_acc': test_acc})

# === 10. 保存 ===
torch.save(model.state_dict(), model_path)
log("Model saved.")

# === 11. グラフ保存 ===
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.grid(); plt.legend(); plt.title('Training Loss')
plt.savefig(loss_plot_path)

plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Acc')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.grid(); plt.legend(); plt.title('Accuracy')
plt.savefig(acc_plot_path)

writer.close()
log("Training complete.")