import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

from src.models.transformer import TimeSeriesTransformer, TimeSeriesDataset

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化模型训练器
        
        Args:
            model: 要训练的模型
            device: 训练设备
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            (平均验证损失, 评估指标字典)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target.unsqueeze(1))
                total_loss += loss.item()
                
                preds = (output > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds),
            'recall': recall_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds)
        }
        
        return total_loss / len(val_loader), metrics
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int,
             learning_rate: float = 0.001,
             save_path: str = None) -> Dict[str, list]:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            n_epochs: 训练轮数
            learning_rate: 学习率
            save_path: 模型保存路径
            
        Returns:
            训练历史记录
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer)
            
            # 验证
            val_loss, metrics = self.validate(val_loader)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(metrics)
            
            # 打印进度
            logger.info(f'Epoch {epoch+1}/{n_epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f}')
            logger.info(f'Val Loss: {val_loss:.4f}')
            logger.info(f'Metrics: {metrics}')
            
            # 保存最佳模型
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f'Best model saved to {save_path}')
        
        return history

def prepare_data(data_path: str,
                seq_length: int,
                batch_size: int,
                train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    准备训练数据
    
    Args:
        data_path: 数据文件路径
        seq_length: 序列长度
        batch_size: 批次大小
        train_ratio: 训练集比例
        
    Returns:
        (训练数据加载器, 验证数据加载器)
    """
    # 加载数据
    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    
    # 准备特征和目标
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd',
        'bb_high', 'bb_low', 'obv'
    ]
    
    X = torch.FloatTensor(df[feature_columns].values)
    y = torch.FloatTensor(df['target'].values)
    
    # 创建数据集
    dataset = TimeSeriesDataset(X, y, seq_length)
    
    # 划分训练集和验证集
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def plot_training_history(history: Dict[str, list], save_path: str = None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史记录
        save_path: 图像保存路径
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制评估指标
    plt.subplot(1, 2, 2)
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        values = [m[metric] for m in history['metrics']]
        plt.plot(values, label=metric.capitalize())
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """主函数"""
    # 设置参数
    data_path = 'data/processed/processed_BTCUSDT_1h_2023-01-01_2024-02-20.csv'
    seq_length = 24
    batch_size = 32
    n_epochs = 50
    learning_rate = 0.001
    
    # 创建模型保存目录
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # 准备数据
    logger.info("准备数据...")
    train_loader, val_loader = prepare_data(
        data_path=data_path,
        seq_length=seq_length,
        batch_size=batch_size
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = TimeSeriesTransformer(
        input_dim=13,  # 特征维度
        d_model=64,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=seq_length
    )
    
    # 训练模型
    logger.info("开始训练...")
    trainer = ModelTrainer(model)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        save_path=model_dir / 'best_model.pth'
    )
    
    # 绘制训练历史
    plot_training_history(
        history,
        save_path=model_dir / 'training_history.png'
    )

if __name__ == "__main__":
    main() 