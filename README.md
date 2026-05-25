# zero-deep-learning
# PyTorch Deep Learning Fundamentals Tutorial / PyTorch 深度学习基础教程

Based on XiaoTuDui PyTorch introductory series.  
源自小土堆 PyTorch 入门教程：  
https://www.bilibili.com/video/BV1hE411t7RN/

This repo reorganizes the key points with clean, minimal code examples.  
本仓库用清晰简洁的代码重新整理重点内容，  
covering dataset handling, model building, training and testing pipelines from scratch.  
从零开始讲解 PyTorch 的数据集处理、模型构建、训练与测试全流程。

---

## 📚 Tutorial Contents / 教程内容

| File / 文件 | Description / 说明 |
|-----------|------------------|
| 01_build_own_dataset.py | Custom Dataset / 自定义数据集 |
| 02_tensorboard.py | TensorBoard Visualization / TensorBoard 可视化 |
| 03_transforms.py | Data Preprocessing & Augmentation / 数据预处理与增强 |
| 04_dataset_in_torchvision.py | Built-in Datasets in torchvision / torchvision 内置数据集 |
| 05_dataloader.py | DataLoader Usage / DataLoader 使用 |
| 06_module.py | Basics of nn.Module / nn.Module 基础 |
| 07_pool.py | Pooling Layers / 池化层 |
| 08_nonlinear_linear.py | Non-linear & Linear Layers / 非线性与线性层 |
| 09_sequential.py | Sequential Container / Sequential 容器 |
| 10_loss_backward_optim.py | Loss Functions, Backpropagation, Optimizers / 损失函数、反向传播、优化器 |
| 11_fix_save_model.py | Model Saving & Loading / 模型保存与加载 |

### 📖 Recommended Learning Order / 建议学习顺序

Please follow the file numbering strictly:  
请严格按照文件编号顺序学习：

**01 → 02 → 03 → ... → 11**

---

## 🚀 Full Implementation / 完整实现

- `model.py` – Model Definition / 模型定义
- `train.py` – Training Script / 训练脚本
- `test.py` – Testing Script / 测试脚本

---

## 📦 Environment Setup / 环境安装

### 1. Install PyTorch / 安装 PyTorch

```bash
# CUDA 11.8：
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU Only: 
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```

### 2. Install Other Dependencies / 安装其他依赖

```bash
pip install -r requirements.txt
```

---

## 🖼️ Dataset Download / 数据集下载

### Ants & Bees Dataset (for 01*.py)  
### 蚂蚁蜜蜂数据集（用于 01*.py）

Download and extract into the `data/` folder:  
下载后解压到 `data/` 目录下：

https://download.pytorch.org/tutorial/hymenoptera_data.zip

---

### CIFAR10 Dataset (used in 04*.py)  
### CIFAR10 数据集（在 04*.py 中使用）

When running CIFAR10-related code for the first time, change:  
首次运行 CIFAR10 相关代码时，请将：

```python
download=False
```

to / 改为：

```python
download=True
```
