import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import time

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
        """
        # 获取K线数据
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=lookback_periods
        )
        
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
            
        return df.set_index('timestamp')
    
    def get_current_price(self, symbol: str) -> float:
        """
        获取当前价格
        
        Args:
            symbol: 交易对符号
            
        Returns:
            当前价格
        """
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: 包含OHLCV数据的DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 计算移动平均线
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # 计算布林带
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_high'] = df['bb_middle'] + (bb_std * 2)
        df['bb_low'] = df['bb_middle'] - (bb_std * 2)
        
        # 计算OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据
        
        Args:
            df: 包含技术指标的DataFrame
            
        Returns:
            处理后的特征DataFrame
        """
        # 选择特征列
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd',
            'bb_high', 'bb_low', 'obv'
        ]
        
        # 删除包含NaN的行
        df = df[feature_columns].dropna()
        
        return df
    
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
        """
        # 获取历史数据
        df = self.get_historical_klines(symbol, interval, lookback_periods)
        
        # 计算技术指标
        df = self.calculate_technical_indicators(df)
        
        # 准备特征
        df = self.prepare_features(df)
        
        return df 