import os
import re
import matplotlib.pyplot as plt

# 出力先ディレクトリ
outputs_path = '/content/drive/MyDrive/Colab Notebooks/NFFA-SEM/outputs'

# 学習曲線用の辞書を初期化
runs_data = {}

for run_name in os.listdir(outputs_path):
    run_path = os.path.join(outputs_path, run_name)
    log_path = os.path.join(run_path, 'train.log')

    if not os.path.isdir(run_path) or not os.path.isfile(log_path):
        continue

    train_loss = []
    train_acc = []
    val_acc = []

    with open(log_path, 'r') as f:
        for line in f:
            # 例: [Epoch 3] Train Loss: 1.2896, Train Acc: 0.5667
            m1 = re.match(r".*\[Epoch (\d+)\] Train Loss: ([\d.]+), Train Acc: ([\d.]+)", line)
            m2 = re.match(r".*\[Epoch (\d+)\] Validation Acc: ([\d.]+)", line)

            if m1:
                train_loss.append(float(m1.group(2)))
                train_acc.append(float(m1.group(3)))
            elif m2:
                val_acc.append(float(m2.group(2)))

    if len(train_loss) > 0:
        runs_data[run_name] = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        }

# ==== グラフ描画 ====

# 1. Loss
plt.figure(figsize=(10, 5))
for run, data in runs_data.items():
    plt.plot(data["train_loss"], label=f'{run} - Loss')
plt.title("Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 2. Accuracy
plt.figure(figsize=(10, 5))
for run, data in runs_data.items():
    plt.plot(data["train_acc"], label=f'{run} - Train Acc')
    if len(data["val_acc"]) > 0:
        plt.plot(data["val_acc"], linestyle='--', label=f'{run} - Val Acc')
plt.title("Train & Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()