import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time
from src.data.preprocess import DataPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataCollector:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """
        初始化币安数据收集器
        
        Args:
            api_key: 币安API密钥（可选）
            api_secret: 币安API密钥（可选）
        """
        self.client = Client(api_key, api_secret)
        self.data_preprocessor = DataPreprocessor()

    def get_historical_klines(self,
                            symbol: str,
                            interval: str,
                            lookback_periods: int = 100) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            lookback_periods: 回溯期数
            
        Returns:
            包含K线数据的DataFrame
            
        Raises:
            ValueError: 当获取的数据为空或数据量不足时
        """
        try:
            # 获取K线数据
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=lookback_periods
            )
            
            if not klines:
                raise ValueError(f"无法获取 {symbol} 的K线数据")
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # 转换数据类型
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # 验证数据量
            if len(df) < lookback_periods:
                logger.warning(f"获取的数据量({len(df)})小于请求的数量({lookback_periods})")
            
            return df.set_index('timestamp')
            
        except Exception as e:
            logger.error(f"获取历史K线数据时发生错误: {str(e)}")
            raise
    
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        Args:
            symbol: 交易对符号
            
        Returns:
            当前价格
            
        Raises:
            ValueError: 当无法获取价格时
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            if not ticker or 'price' not in ticker:
                raise ValueError(f"无法获取 {symbol} 的当前价格")
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"获取当前价格时发生错误: {str(e)}")
            raise
    
    
    def collect_realtime_data(self,
                            symbol: str,
                            interval: str,
                            lookback_periods: int = 100) -> pd.DataFrame:
        """
        收集实时数据并处理
        
        Args:
            symbol: 交易对符号
            interval: K线间隔
            lookback_periods: 回溯期数
            
        Returns:
            处理后的特征DataFrame
            
        Raises:
            ValueError: 当数据收集或处理失败时
        """
        try:
            # 获取历史数据
            df = self.get_historical_klines(symbol, interval, lookback_periods)
            
            # 计算技术指标
            df = self.data_preprocessor.add_technical_indicators(df)

            # 处理缺失值
            df = self.data_preprocessor.handle_missing_values(df)

            # 标准化特征
            scaler = self.data_preprocessor.load_scaler(self.data_preprocessor.scaler_file)
            df = self.data_preprocessor.transform_features(df, self.data_preprocessor.feature_columns, scaler)

            # 准备特征
            df = df[self.data_preprocessor.feature_columns]

            
            logger.info(f"成功收集并处理了{len(df)}个数据点")
            return df
            
        except Exception as e:
            logger.error(f"收集实时数据时发生错误: {str(e)}")
            raise 