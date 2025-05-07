# AI量化交易系统

这是一个基于深度学习的加密货币量化交易系统，使用Transformer模型进行价格预测。

## 项目结构

```
.
├── data/                   # 数据存储目录
│   ├── raw/               # 原始数据
│   └── processed/         # 处理后的数据
├── models/                # 模型存储目录
├── src/                   # 源代码
│   ├── data/             # 数据处理相关代码
│   ├── features/         # 特征工程相关代码
│   ├── models/           # 模型定义相关代码
│   ├── training/         # 模型训练相关代码
│   └── utils/            # 工具函数
├── notebooks/            # Jupyter notebooks
├── configs/              # 配置文件
├── requirements.txt      # 项目依赖
└── README.md            # 项目说明
```

## 功能特点

- 币安数据实时抓取
- 数据预处理和特征工程
- 基于Transformer的深度学习模型
- 模型训练和评估
- 回测系统
- 性能指标计算

## 安装

1. 克隆项目
```bash
git clone [项目地址]
cd [项目目录]
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境变量
创建`.env`文件并添加币安API密钥：
```
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

## 使用方法

1. 数据获取
```bash
python src/data/fetch_data.py
```

2. 数据预处理
```bash
python src/data/preprocess.py
```

3. 模型训练
```bash
python src/training/train.py
```

4. 模型评估
```bash
python src/training/evaluate.py
```

## 注意事项

- 请确保在使用前配置好币安API密钥
- 建议在虚拟环境中运行项目
- 数据获取可能需要一定时间，请耐心等待 