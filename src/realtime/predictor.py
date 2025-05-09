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
from configs.config import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealtimePredictor:
    def __init__(self,
                 model_path: str,
                 seq_length: int = None,
                 prediction_horizon: int = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化实时预测器
        
        Args:
            model_path: 模型权重文件路径
            seq_length: 序列长度（如果为None，则使用配置文件中的值）
            prediction_horizon: 预测时间跨度（小时）（如果为None，则使用配置文件中的值）
            device: 运行设备
        """
        self.seq_length = seq_length or config.data.seq_length
        self.prediction_horizon = prediction_horizon or config.data.target_period
        self.device = device

        self.model = TimeSeriesTransformer(
            input_dim=len(config.data.feature_columns),
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_encoder_layers=config.model.num_encoder_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            max_seq_length=config.model.max_seq_length
        )
        
        # 加载模型权重
        state_dict = torch.load(model_path)

        self.model.load_state_dict(state_dict)
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
            seq_length: 序列长度
            
        Returns:
            预测结果字典
            
        Raises:
            ValueError: 当数据不足或处理失败时
        """
        try:
            lookback_periods = self.seq_length + 100
            # 收集数据
            df = self.data_collector.collect_realtime_data(symbol, interval, lookback_periods)
            
            # 准备输入数据
            input_data = self.prepare_input_data(df)
            
            # 获取当前价格
            current_price = self.data_collector.get_current_price(symbol)
            
            # 进行预测
            with torch.no_grad():
                output = self.model(input_data)
                prediction = (output > 0.5).float().item()
                confidence = output.item() if output > 0.5 else 1-output.item()
            
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
                'is_correct': None,
            }
            
            # 保存预测结果
            self._save_prediction(result)
            
            return result
            
        except ValueError as e:
            logger.error(f"数据验证失败: {str(e)}")
            raise
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