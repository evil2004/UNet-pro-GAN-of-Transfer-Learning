# GAN 模型微调项目

本项目包含使用 PyTorch 实现的 GAN 模型微调代码。

## 环境要求

- Python 3.x
- PyTorch >= 1.10.0
- torchvision == 0.11.1
- Pillow == 9.0.1
- matplotlib == 3.5.1
- scikit-learn == 1.0.2
- tqdm == 4.62.3
- pandas == 1.3.5
- numpy == 1.21.5
- opencv-python == 4.5.5.64

## 安装依赖

使用 pip 安装所需的依赖项：

```bash
pip install -r requirements.txt
```

## 数据准备

将数据集放置在 `data` 目录下。目录结构应类似于：

```
data/
├── train/
│   ├── classA/
│   │   ├── image1.jpg
│   │   └── ...
│   └── classB/
│       ├── image1.jpg
│       └── ...
└── test/
    ├── classA/
    │   ├── image1.jpg
    │   └── ...
    └── classB/
        ├── image1.jpg
        └── ...
```

## 运行模型微调

使用以下命令运行模型微调：

```bash
python run_finetune.py [options]
```

可选参数:

- `--data_root`: 数据集根目录 (默认为 `data`)
- `--results_dir`: 结果保存目录 (默认为 `results_finetune`)
- `--models_dir`: 模型保存目录 (默认为 `models`)
- `--batch_size`: 批次大小 (默认为 `16`)
- `--epochs`: 源模型训练轮次 (默认为 `100`)
- `--finetune_epochs`: 微调轮次 (默认为 `50`)
- `--lr`: 学习率 (默认为 `0.0002`)

例如:

```bash
python run_finetune.py --finetune_epochs 30 --lr 0.0001
```

训练完成后，微调后的模型将保存在 `--models_dir`/finetune 目录下。

## 运行模型测试 (微调后)

使用以下命令测试微调后的模型：

```bash
python run_finetune_test.py [options]
```

可选参数:

- `--data_root`: 数据集根目录 (默认为 `data`)
- `--results_dir`: 测试结果保存目录 (默认为 `finetune_test_results`)
- `--source_models_dir`: 源模型目录 (默认为 `models/finetune`)
- `--finetune_models_dir`: 微调模型目录 (默认为 `models/finetune`)
- `--batch_size`: 批次大小 (默认为 `32`)

例如:

```bash
python run_finetune_test.py --batch_size 64
```

测试结果将保存在 `--results_dir` 目录下，包括可视化报告。

## 其他脚本

- `model_gan.py`: GAN 模型结构定义。
- `finetune_model.py`: 实际执行微调的核心逻辑。
- `test_finetune_model.py`: 实际执行微调测试的核心逻辑。
- `test_model.py`: 用于测试非微调模型的脚本（如果适用）。
- `visualization.py`: 包含用于生成可视化结果的函数。

## 注意

- 请确保数据集路径和模型保存路径正确设置。
- 根据您的硬件配置调整批次大小。 