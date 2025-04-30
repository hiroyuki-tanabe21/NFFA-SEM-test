import os
import itertools
import subprocess
import yaml

def load_base_config(base_config_path):
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    base_config_path = '/content/drive/MyDrive/Colab Notebooks/NFFA-SEM/configs/config.yaml'

    # 探索するハイパーパラメータ
    learning_rates = [0.01, 0.001, 0.0001] 
    batch_sizes = [32] #[32,64,128]

    # base設定を読み込む
    base_config = load_base_config(base_config_path)

    # 全組み合わせを生成
    for lr, bs in itertools.product(learning_rates, batch_sizes):
        print(f"Running with lr={lr}, batch_size={bs}")

        # コマンドを組み立てる
        cmd = [
            "python", "/content/drive/MyDrive/Colab Notebooks/NFFA-SEM/scripts/train.py",
            "--config", base_config_path,
            "--lr", str(lr),
            "--batch_size", str(bs)
        ]

        # 実行
        subprocess.run(cmd)

if __name__ == "__main__":
    main()