import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional
import json
from pathlib import Path

from src.models.transformer import TimeSeriesTransformer
from src.realtime.data_collector import BinanceDataCollector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimePredictor:
    def __init__(self,
                 model_path: str,
                 seq_length: int = 24,
                 prediction_horizon: int = 1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化实时预测器
        
        Args:
            model_path: 模型权重文件路径
            seq_length: 序列长度
            prediction_horizon: 预测时间跨度（小时）
            device: 运行设备
        """
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        # 加载模型
        self.model = TimeSeriesTransformer(
            input_dim=13,
            d_model=64,
            nhead=8,
            num_encoder_layers=3,
            dim_feedforward=256,
            dropout=0.1,
            max_seq_length=seq_length
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(device)
        self.model.eval()
        
        # 初始化数据收集器
        self.data_collector = BinanceDataCollector()
        
        # 创建预测结果存储目录
        self.results_dir = Path('data/predictions')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_input_data(self, df: pd.DataFrame) -> torch.Tensor:
        """
        准备模型输入数据
        
        Args:
            df: 特征DataFrame
            
        Returns:
            模型输入张量
        """
        # 获取最新的seq_length个数据点
        data = df.iloc[-self.seq_length:].values
        return torch.FloatTensor(data).unsqueeze(0).to(self.device)
    
    def make_prediction(self, symbol: str, interval: str = '1h') -> Dict:
        """
        进行预测
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            
        Returns:
            预测结果字典
        """
        try:
            # 收集数据
            df = self.data_collector.collect_realtime_data(symbol, interval)
            
            # 准备输入数据
            input_data = self.prepare_input_data(df)
            
            # 获取当前价格
            current_price = self.data_collector.get_current_price(symbol)
            
            # 进行预测
            with torch.no_grad():
                output = self.model(input_data)
                prediction = (output > 0.5).float().item()
                confidence = output.item()
            
            # 记录预测时间
            prediction_time = datetime.now()
            
            # 创建预测结果
            result = {
                'symbol': symbol,
                'prediction_time': prediction_time.isoformat(),
                'current_price': current_price,
                'prediction': int(prediction),
                'confidence': float(confidence),
                'prediction_horizon': self.prediction_horizon,
                'verified': False,
                'actual_direction': None,
                'is_correct': None
            }
            
            # 保存预测结果
            self._save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"预测过程中发生错误: {str(e)}")
            raise
    
    def verify_prediction(self, prediction_id: str) -> Dict:
        """
        验证预测结果
        
        Args:
            prediction_id: 预测ID
            
        Returns:
            更新后的预测结果字典
        """
        try:
            # 加载预测结果
            result = self._load_prediction(prediction_id)
            
            # 获取当前价格
            current_price = self.data_collector.get_current_price(result['symbol'])
            
            # 计算实际价格变化方向
            price_change = current_price - result['current_price']
            actual_direction = 1 if price_change > 0 else 0
            
            # 更新预测结果
            result['verified'] = True
            result['actual_direction'] = actual_direction
            result['is_correct'] = result['prediction'] == actual_direction
            
            # 保存更新后的结果
            self._save_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"验证预测结果时发生错误: {str(e)}")
            raise
    
    def _save_prediction(self, result: Dict):
        """
        保存预测结果
        
        Args:
            result: 预测结果字典
        """
        prediction_id = f"{result['symbol']}_{result['prediction_time']}"
        file_path = self.results_dir / f"{prediction_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=4)
    
    def _load_prediction(self, prediction_id: str) -> Dict:
        """
        加载预测结果
        
        Args:
            prediction_id: 预测ID
            
        Returns:
            预测结果字典
        """
        file_path = self.results_dir / f"{prediction_id}.json"
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def get_prediction_stats(self) -> Dict:
        """
        获取预测统计信息
        
        Returns:
            统计信息字典
        """
        predictions = []
        for file_path in self.results_dir.glob('*.json'):
            with open(file_path, 'r') as f:
                predictions.append(json.load(f))
        
        # 只统计已验证的预测
        verified_predictions = [p for p in predictions if p['verified']]
        
        if not verified_predictions:
            return {
                'total_predictions': 0,
                'verified_predictions': 0,
                'accuracy': 0.0,
                'correct_predictions': 0,
                'incorrect_predictions': 0
            }
        
        correct_predictions = sum(1 for p in verified_predictions if p['is_correct'])
        
        return {
            'total_predictions': len(predictions),
            'verified_predictions': len(verified_predictions),
            'accuracy': correct_predictions / len(verified_predictions),
            'correct_predictions': correct_predictions,
            'incorrect_predictions': len(verified_predictions) - correct_predictions
        } 