import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import ta
from sklearn.preprocessing import StandardScaler
import joblib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, data_dir: str = 'data'):
        """
        初始化数据预处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据"""
        return pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # 添加趋势指标
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # 添加动量指标
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        
        # 添加波动率指标
        df['bb_high'] = ta.volatility.bollinger_hband(df['close'])
        df['bb_low'] = ta.volatility.bollinger_lband(df['close'])
        
        # 添加成交量指标
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        return df
    
    def create_target(self, df: pd.DataFrame, target_period: int = 24) -> pd.DataFrame:
        """
        创建目标变量（未来价格变化）
        
        Args:
            df: 原始数据DataFrame
            target_period: 预测周期
            
        Returns:
            添加了目标变量的DataFrame
        """
        # 计算未来价格变化百分比
        df['future_return'] = df['close'].shift(-target_period) / df['close'] - 1
        
        # 创建分类标签（1: 上涨, 0: 下跌）
        df['target'] = (df['future_return'] > 0).astype(int)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 删除包含缺失值的行
        df = df.dropna()
        return df
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        特征标准化
        
        Args:
            df: 原始数据DataFrame
            feature_columns: 需要标准化的特征列名列表
            
        Returns:
            标准化后的DataFrame和scaler对象
        """
        scaler = StandardScaler()
        df[feature_columns] = scaler.fit_transform(df[feature_columns])
        return df, scaler
    
    def prepare_data(self, input_file: str, target_period: int = 24) -> None:
        """
        准备训练数据
        
        Args:
            input_file: 输入文件路径
            target_period: 预测周期
        """
        try:
            # 加载数据
            logger.info(f"加载数据: {input_file}")
            df = self.load_data(input_file)
            
            # 添加技术指标
            logger.info("添加技术指标...")
            df = self.add_technical_indicators(df)
            
            # 创建目标变量
            logger.info("创建目标变量...")
            df = self.create_target(df, target_period)
            
            # 处理缺失值
            logger.info("处理缺失值...")
            df = self.handle_missing_values(df)
            
            # 定义特征列
            feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd',
                'bb_high', 'bb_low', 'obv'
            ]
            
            # 标准化特征
            logger.info("标准化特征...")
            df, scaler = self.normalize_features(df, feature_columns)
            
            # 保存处理后的数据
            output_file = self.processed_dir / f"processed_{Path(input_file).name}"
            df.to_csv(output_file)
            logger.info(f"处理后的数据已保存到: {output_file}")
            
            # 保存scaler
            scaler_file = self.processed_dir / 'scaler.joblib'
            joblib.dump(scaler, scaler_file)
            logger.info(f"Scaler已保存到: {scaler_file}")
            
        except Exception as e:
            logger.error(f"数据预处理过程中发生错误: {str(e)}")
            raise

def main():
    """主函数"""
    preprocessor = DataPreprocessor()
    
    # 处理所有原始数据文件
    raw_files = list(Path('data/raw').glob('*.csv'))
    
    for file in raw_files:
        logger.info(f"处理文件: {file}")
        preprocessor.prepare_data(str(file))

if __name__ == "__main__":
    main() 