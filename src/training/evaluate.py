import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
import seaborn as sns

from src.models.transformer import TimeSeriesTransformer, TimeSeriesDataset
from src.training.train import prepare_data

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化模型评估器
        
        Args:
            model: 要评估的模型
            device: 评估设备
        """
        self.model = model.to(device)
        self.device = device
        
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            (评估指标字典, 预测概率, 真实标签)
        """
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probs = output.cpu().numpy()
                preds = (output > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.cpu().numpy())
        
        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds),
            'recall': recall_score(all_targets, all_preds),
            'f1': f1_score(all_targets, all_preds)
        }
        
        return metrics, np.array(all_probs), np.array(all_targets)
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 图像保存路径
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, save_path: str = None):
        """
        绘制ROC曲线
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 图像保存路径
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def main():
    """主函数"""
    # 设置参数
    data_path = 'data/processed/processed_BTCUSDT_1h_2023-01-01_2024-02-20.csv'
    model_path = 'models/best_model.pth'
    seq_length = 24
    batch_size = 32
    
    # 创建评估结果保存目录
    eval_dir = Path('models/evaluation')
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备数据
    logger.info("准备数据...")
    _, val_loader = prepare_data(
        data_path=data_path,
        seq_length=seq_length,
        batch_size=batch_size
    )
    
    # 创建模型
    logger.info("加载模型...")
    model = TimeSeriesTransformer(
        input_dim=13,
        d_model=64,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_length=seq_length
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(model_path))
    
    # 评估模型
    logger.info("开始评估...")
    evaluator = ModelEvaluator(model)
    metrics, probs, targets = evaluator.evaluate(val_loader)
    
    # 打印评估指标
    logger.info("评估指标:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(
        targets,
        (probs > 0.5).astype(int),
        save_path=eval_dir / 'confusion_matrix.png'
    )
    
    # 绘制ROC曲线
    evaluator.plot_roc_curve(
        targets,
        probs,
        save_path=eval_dir / 'roc_curve.png'
    )

if __name__ == "__main__":
    main() 