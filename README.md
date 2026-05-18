# zero-deep-learning
# PyTorch 深度学习基础教程

源自小土堆 Pytorch 入门教程，  
https://www.bilibili.com/video/BV1hE411t7RN/  

整理重点以清晰的代码来展示。  
从零开始的 PyTorch 学习教程，涵盖数据集处理、模型构建、训练与测试全流程。

## 📚 教程内容

| 文件 | 内容 |
|------|------|
| 01_build_own_dataset.py | 自定义数据集 |
| 02_tensorboard.py | TensorBoard 可视化 |
| 03_transforms.py | 数据预处理与增强 |
| 04_dataset_in_torchvision.py | torchvision 内置数据集 |
| 05_dataloader.py | DataLoader 使用 |
| 06_module.py | nn.Module 基础 |
| 07_pool.py | 池化层 |
| 08_nonlinear_linear.py | 非线性与线性层 |
| 09_sequential.py | Sequential 容器 |
| 10_loss_backward_optim.py | 损失函数、反向传播、优化器 |
| 11_fix_save_model.py | 模型保存与加载 |

### 📖 学习顺序（重要）

建议按文件编号 **01 → 02 → 03 → ... → 11** 顺序学习

## 🚀 完整实现

- `model.py` - 模型定义
- `train.py` - 训练脚本
- `test.py` - 测试脚本

## 📦 环境安装

### 1. 安装 PyTorch

CUDA 11.8：

```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU Only
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```

### 2. 安装其它依赖

```bash
pip install -r requirements.txt
```

## 🖼️ 数据集下载

### 蚂蚁蜜蜂数据集（用于 01*.py）

下载到 data 文件夹下并解压：

https://download.pytorch.org/tutorial/hymenoptera_data.zip

### CIFAR10 数据集（在 04*.py 处下载）

首次运行 CIFAR10 相关代码 04*.py 时，请将：

```python
download=False
```

改为：

```python
download=True
```
