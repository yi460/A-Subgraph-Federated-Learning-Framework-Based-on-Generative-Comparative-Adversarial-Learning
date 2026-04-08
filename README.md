
# 联邦图学习：基于 GAN 与对比学习的 Non-IID 数据优化框架

这是一个使用 PyTorch 和 PyTorch Geometric 实现的联邦图神经网络（Federated Graph Neural Networks）项目。针对联邦学习中广泛存在的 **Non-IID（非独立同分布）** 数据孤岛问题，本项目提出了一种基于生成对抗网络（GAN）和对比学习（Contrastive Learning）的全局模型优化方案。

代码在三个经典的图节点分类基准数据集（**Cora**, **Citeseer**, **PubMed**）上进行了实验，并与基线模型（FedAvg）及其他 SOTA 方法进行了对比。

##  核心特性

* **狄利克雷 Non-IID 数据划分**：内置基于狄利克雷分布（Dirichlet Distribution，$\alpha=0.5$）的异构数据划分机制，模拟真实的联邦学习客户端数据分布不均场景。
* **基于原型的知识抽取（Prototype Extraction）**：客户端使用 GCN 提取图节点特征，并计算每个类别的特征原型（Prototypes），在不泄露原始图数据的前提下上传至服务器。
* **服务器端 GAN 数据增强**：服务器端训练一个生成器（Generator）和判别器（Discriminator），利用收集到的真实原型生成高质量的“伪原型（Fake Prototypes）”，以弥补全局数据分布的缺失。
* **对比学习（Contrastive Learning）**：在生成过程中引入对比损失，拉近同类原型的距离，推开不同类原型的距离，防止训练初期震荡。
* **柔性微调（Soft Update）**：服务器利用混合数据（真实原型 + 生成原型）对分类器进行微调，并通过 $10\%$ 的动量（Alpha）柔性更新到全局模型权重中。
* **一键式综合可视化**：运行结束后自动生成包含数据分布热力图、收敛曲线对比、Loss 动态曲线、t-SNE 降维聚类图以及 SOTA 性能对比柱状图的综合面板。

## 文件结构

项目中包含三个主要的可执行 Python 脚本，分别对应三个不同的基准数据集：

* `1.py`：在 **Cora** 数据集上运行实验。
* `2.py`：在 **Citeseer** 数据集上运行实验。
* `3.py`：在 **PubMed** 数据集上运行实验。

## 依赖环境

在运行代码之前，请确保您的环境中安装了以下 Python 库。推荐使用 Python 3.8 及以上版本。

```bash
# 核心深度学习库
pip install torch
pip install torch_geometric

# 数据处理与可视化库
pip install numpy matplotlib seaborn scikit-learn
```

*注：代码会自动检测并使用 GPU（CUDA），如果未配置 CUDA 则会自动回退至 CPU 运行。由于访问网络原因，代码中 Planetoid 数据源已配置为 Gitee 镜像 (`https://gitee.com/jiajiewu/planetoid/raw/master/data`) 以加速下载。*

## 快速开始

克隆或下载本仓库后，直接运行对应的 Python 脚本即可启动整个联邦训练及可视化流程。

**以 Cora 数据集为例：**

```bash
python 1.py
```

**运行流程说明：**
1.  **数据下载与划分**：代码会自动下载数据集到 `./data` 目录下，并按设定的 $\alpha$ 值分配给 5 个客户端。
2.  **基线训练 (FedAvg)**：首先会运行标准的 FedAvg 算法作为对比基线。
3.  **核心算法训练 (Ours)**：接着运行带有 GAN 和对比学习的改进版联邦学习。前 10 轮为**预热期（Warmup Phase）**，随后开启生成器训练与全局柔性微调。
4.  **生成图表**：终端训练完成后，会自动弹出一个综合图表（Matplotlib），展示详细的实验对比结果。

## 超参数配置

如果您想调整实验设置，可以直接在每个 `.py` 文件的 `Args` 类中修改相关参数：

```python
class Args:
    num_clients = 5          # 客户端数量
    num_rounds = 40          # 联邦学习总通讯轮数
    warmup_rounds = 10       # 预热轮数（不开启GAN微调）
    local_epochs = 3         # 客户端本地训练 Epoch 数
    server_gan_epochs = 20   # 服务器端 GAN 训练 Epoch 数
    alpha = 0.5              # 狄利克雷分布参数（越小 Non-IID 程度越高）
    hidden_dim = 64          # GCN 隐藏层维度
    latent_dim = 32          # 原型/潜变量维度
    lr_gcn = 0.01            # 客户端 GCN 学习率
    lr_gan = 0.001           # 服务器 GAN 学习率
```

## 可视化结果说明

运行结束后展示的图表包含以下 5 个子图：
1.  **Client Data Distribution**：热力图展示不同客户端上各个类别的样本数量，直观体现 Non-IID 分布。
2.  **Convergence Comparison**：展示我们提出的方法（红线）与 FedAvg 基线（灰虚线）在测试集上的准确率收敛对比。
3.  **Training Loss Dynamics**：展示生成器 Loss 与对比 Loss 的变化趋势（仅在预热期之后）。
4.  **t-SNE 聚类**：将真实的特征原型（圆形）与 GAN 生成的伪特征原型（星形）降维至 2D 展示其分布。
5.  **SOTA Comparison**：柱状图对比本方案与现存 SOTA 方案的最终准确率。
