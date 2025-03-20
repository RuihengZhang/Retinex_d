# Retinex_D - 图像增强工具包

RetinexD 是一个基于Retinex理论的图像增强框架，项目核心是提供了一种适用于红外图像的Retinex分解方法，代码文件为Decomposition/uretinex.py。

## 其它贡献

- **图像去噪**
  - 搜集了大量图像去噪方法，在文件夹Denoising下
- **图像增强**
  - 搜集了大量暗光增强方法，提升红外图像的对比度，在文件夹Lightening
- **图像预处理**
  - 支持添加多种噪声：高斯噪声、椒盐噪声、泊松噪声
  - 支持对比度调整

## 目录结构

```
RetinexD/
├── Checkpoints/          # 模型检查点
├── Datasets/             # 数据集
├── Decomposition/        # 图像分解算法
├── Denoising/            # 去噪算法
├── Lightening/           # 图像增强算法
├── Metrics/              # 评估指标
└── Script/               # 主要脚本
    ├── prepration.py     # 图像预处理
    ├── denoising.py      # 图像去噪
    └── lightening.py     # 图像增强
```

## 安装说明

1. 克隆本仓库：
   ```bash
   git clone https://github.com/yourusername/RetinexD.git
   cd RetinexD
   ```

2. 安装依赖：
    pytorch框架，使用的第三方库均为cv领域常见安装包

## 使用说明

### 图像预处理
```python
# 添加噪声
python Script/prepration.py

# 调整对比度
python Script/prepration.py --contrast
```

### 图像去噪
```python
# 使用X-Net去噪
python Script/denoising.py
```

### 图像增强
```python
# 使用gamma校正
python Script/lightening.py
```

## 示例

Script目录下的其它文件为不同实验的对比实验，详情见论文

## 贡献指南

欢迎提交issue和pull request。请确保：
1. 代码符合PEP8规范
2. 添加必要的单元测试
3. 更新相关文档
