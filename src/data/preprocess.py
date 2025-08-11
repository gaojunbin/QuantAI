import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import ta
from sklearn.preprocessing import StandardScaler
import joblib
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """数据预处理器
    
    负责数据的加载、清洗、特征工程和标准化等预处理工作
    """
    def __init__(self, data_dir: str = 'data'):
        """初始化数据预处理器
        
        Args:
            data_dir: 数据目录路径，默认为'data'
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_file = self.processed_dir / 'scaler.joblib'
        
        # 定义所有需要计算的技术指标
        self.feature_columns = [
            # 基础价格和成交量数据
            'open', 'high', 'low', 'close', 'volume',
            
            # 移动平均线指标
            'sma_20', 'sma_50', 'sma_200',  # 简单移动平均线
            'ema_20', 'ema_50', 'ema_200',  # 指数移动平均线
            
            # 动量指标
            'rsi', 'rsi_14', 'rsi_21',      # 相对强弱指标，不同周期
            'macd', 'macd_signal', 'macd_hist',  # MACD指标及其信号
            
            # 波动率指标
            'bb_high', 'bb_low', 'bb_mid',  # 布林带
            'atr', 'adx',                   # 平均真实范围和趋势强度
            
            # 成交量指标
            'obv', 'vwap',                  # 能量潮和成交量加权平均价格
            
            # 随机指标
            'stoch_k', 'stoch_d'            # 随机指标
        ]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载原始数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的DataFrame，索引为时间戳
        """
        return pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标
        
        计算各种技术分析指标，包括趋势、动量、波动率和成交量指标
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            添加了技术指标的DataFrame
        """
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # 添加趋势指标
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)  # 20周期简单移动平均线
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)  # 50周期简单移动平均线
        df['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)  # 200周期简单移动平均线
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)  # 20周期指数移动平均线
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)  # 50周期指数移动平均线
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)  # 200周期指数移动平均线
        
        # 添加高级趋势指标
        df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])  # 一目均衡图A线
        df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])  # 一目均衡图B线
        df['ichimoku_base'] = ta.trend.ichimoku_base_line(df['high'], df['low'])  # 基准线
        df['ichimoku_conv'] = ta.trend.ichimoku_conversion_line(df['high'], df['low'])  # 转换线
        
        # 添加动量指标
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)  # 14周期RSI
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)  # 14周期RSI
        df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)  # 21周期RSI
        
        # 添加高级动量指标
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])  # 资金流量指标
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])  # 顺势指标
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])  # 威廉指标
        
        # MACD指标
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()  # MACD线
        df['macd_signal'] = macd.macd_signal()  # 信号线
        df['macd_hist'] = macd.macd_diff()  # MACD柱状图
        
        # 布林带
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()  # 上轨
        df['bb_low'] = bollinger.bollinger_lband()  # 下轨
        df['bb_mid'] = bollinger.bollinger_mavg()  # 中轨
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']  # 布林带宽度
        df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])  # 布林带百分比
        
        # 成交量指标
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])  # 能量潮
        df['vwap'] = ta.volume.volume_weighted_average_price(  # 成交量加权平均价格
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )
        
        # 添加高级成交量指标
        df['cmf'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])  # 蔡金货币流量
        df['efi'] = ta.volume.force_index(df['close'], df['volume'])  # 强力指数
        df['eom'] = ta.volume.ease_of_movement(df['high'], df['low'], df['volume'])  # 简易波动指标
        
        # 波动率指标
        df['atr'] = ta.volatility.average_true_range(  # 平均真实范围
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['adx'] = ta.trend.adx(  # 平均趋向指标
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        
        # 添加高级波动率指标
        df['natr'] = ta.volatility.normalized_average_true_range(df['high'], df['low'], df['close'])  # 归一化ATR
        df['tr'] = ta.volatility.true_range(df['high'], df['low'], df['close'])  # 真实范围
        df['ui'] = ta.volatility.ulcer_index(df['close'])  # 溃疡指数
        
        # 随机指标
        stoch = ta.momentum.StochasticOscillator(  # 随机指标
            high=df['high'],
            low=df['low'],
            close=df['close']
        )
        df['stoch_k'] = stoch.stoch()  # %K线
        df['stoch_d'] = stoch.stoch_signal()  # %D线
        
        # 添加价格变化特征
        df['price_change'] = df['close'].pct_change()  # 价格变化率
        df['volume_change'] = df['volume'].pct_change()  # 成交量变化率
        
        # 添加价格波动特征
        df['high_low_ratio'] = df['high'] / df['low']  # 高低价比率
        df['close_open_ratio'] = df['close'] / df['open']  # 收盘开盘价比率
        
        # 添加移动平均线交叉特征
        df['sma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)  # SMA交叉信号
        df['ema_cross'] = (df['ema_20'] > df['ema_50']).astype(int)  # EMA交叉信号
        
        # 添加价格动量特征
        df['price_momentum_1'] = df['close'].pct_change(periods=1)  # 1周期动量
        df['price_momentum_3'] = df['close'].pct_change(periods=3)  # 3周期动量
        df['price_momentum_5'] = df['close'].pct_change(periods=5)  # 5周期动量
        
        # 添加波动率特征
        df['volatility_1'] = df['price_change'].rolling(window=1).std()  # 1周期波动率
        df['volatility_3'] = df['price_change'].rolling(window=3).std()  # 3周期波动率
        df['volatility_5'] = df['price_change'].rolling(window=5).std()  # 5周期波动率
        
        # 添加趋势强度特征
        df['trend_strength_1'] = abs(df['price_momentum_1']) / df['volatility_1']  # 1周期趋势强度
        df['trend_strength_3'] = abs(df['price_momentum_3']) / df['volatility_3']  # 3周期趋势强度
        df['trend_strength_5'] = abs(df['price_momentum_5']) / df['volatility_5']  # 5周期趋势强度
        
        # 添加价格波动范围特征
        df['range_1'] = (df['high'].rolling(window=1).max() - df['low'].rolling(window=1).min()) / df['close']  # 1周期波动范围
        df['range_3'] = (df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()) / df['close']  # 3周期波动范围
        df['range_5'] = (df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()) / df['close']  # 5周期波动范围
        
        return df
    
    def create_target(self, df: pd.DataFrame, target_period: int = 12) -> pd.DataFrame:
        """创建目标变量
        
        计算未来价格、价格变化、波动率等目标变量
        
        Args:
            df: 原始数据DataFrame
            target_period: 预测周期
            
        Returns:
            添加了目标变量的DataFrame
        """
        # 计算未来价格
        df['future_price'] = df['close'].shift(-target_period)
        
        # 计算价格变化百分比
        df['price_change_pct'] = (df['future_price'] - df['close']) / df['close']
        
        # 计算波动率
        df['future_volatility'] = df['price_change_pct'].rolling(window=target_period).std()
        
        # 计算价格趋势强度
        df['trend_strength'] = abs(df['price_change_pct']) / df['future_volatility']
        
        # 计算价格动量
        df['price_momentum'] = df['close'].pct_change(periods=target_period)
        
        # 计算价格加速度
        df['price_acceleration'] = df['price_momentum'].diff()
        
        # 计算价格波动范围
        df['price_range'] = (df['high'].rolling(window=target_period).max() - 
                           df['low'].rolling(window=target_period).min()) / df['close']
        
        # 计算未来最高价和最低价
        df['future_high'] = df['high'].shift(-target_period)
        df['future_low'] = df['low'].shift(-target_period)
        
        # 计算未来价格波动范围
        df['future_range'] = (df['future_high'] - df['future_low']) / df['close']
        
        # 计算未来价格趋势
        df['future_trend'] = np.where(df['future_price'] > df['close'], 1, -1)  # 1表示上涨，-1表示下跌
        
        # 计算未来价格波动方向
        df['future_direction'] = np.where(df['price_change_pct'] > 0, 1, 0)  # 1表示上涨，0表示下跌
        
        # 计算未来价格波动幅度分类
        df['future_magnitude'] = pd.qcut(abs(df['price_change_pct']), q=5, labels=[1, 2, 3, 4, 5])  # 将波动幅度分为5个等级
        
        # 计算未来价格波动率分类
        df['future_volatility_class'] = pd.qcut(df['future_volatility'], q=5, labels=[1, 2, 3, 4, 5])  # 将波动率分为5个等级
        
        # 计算未来价格趋势强度分类
        df['future_trend_strength_class'] = pd.qcut(df['trend_strength'], q=5, labels=[1, 2, 3, 4, 5])  # 将趋势强度分为5个等级
        
        # 计算未来价格波动范围分类
        df['future_range_class'] = pd.qcut(df['future_range'], q=5, labels=[1, 2, 3, 4, 5])  # 将波动范围分为5个等级
        
        # 计算未来价格波动特征
        df['future_price_std'] = df['close'].rolling(window=target_period).std().shift(-target_period)  # 未来价格标准差
        df['future_price_mean'] = df['close'].rolling(window=target_period).mean().shift(-target_period)  # 未来价格均值
        df['future_price_skew'] = df['close'].rolling(window=target_period).skew().shift(-target_period)  # 未来价格偏度
        df['future_price_kurt'] = df['close'].rolling(window=target_period).kurt().shift(-target_period)  # 未来价格峰度
        
        # 计算未来价格波动率特征
        df['future_volatility_std'] = df['future_volatility'].rolling(window=target_period).std()  # 未来波动率标准差
        df['future_volatility_mean'] = df['future_volatility'].rolling(window=target_period).mean()  # 未来波动率均值
        df['future_volatility_skew'] = df['future_volatility'].rolling(window=target_period).skew()  # 未来波动率偏度
        df['future_volatility_kurt'] = df['future_volatility'].rolling(window=target_period).kurt()  # 未来波动率峰度
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值
        
        使用多种方法处理缺失值，确保数据的完整性
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 使用前向填充处理缺失值
        df = df.fillna(method='ffill')
        
        # 对于剩余的缺失值，使用后向填充
        df = df.fillna(method='bfill')
        
        # 如果还有缺失值，使用0填充
        df = df.fillna(0)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
        """特征标准化
        
        对每个特征分别进行标准化，处理异常值
        
        Args:
            df: 原始数据DataFrame
            feature_columns: 需要标准化的特征列名列表
            
        Returns:
            标准化后的DataFrame和scaler对象
        """
        # 使用稳健缩放器处理异常值
        scaler = StandardScaler()
        
        # 对每个特征分别进行标准化
        for col in feature_columns:
            if col in df.columns:
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
        
        return df, scaler

    def save_scaler(self, scaler: StandardScaler, save_path: str):
        """保存scaler对象
        
        Args:
            scaler: 要保存的StandardScaler对象
            save_path: 保存路径
        """
        joblib.dump(scaler, save_path)
        
    def load_scaler(self, load_path: str) -> StandardScaler:
        """加载scaler对象
        
        Args:
            load_path: scaler文件路径
            
        Returns:
            加载的StandardScaler对象
        """
        return joblib.load(load_path)
        
    def transform_features(self, df: pd.DataFrame, feature_columns: List[str], scaler: StandardScaler) -> pd.DataFrame:
        """使用已有的scaler转换特征
        
        Args:
            df: 原始数据DataFrame
            feature_columns: 需要标准化的特征列名列表
            scaler: 已训练好的StandardScaler对象
            
        Returns:
            标准化后的DataFrame
        """
        df[feature_columns] = scaler.transform(df[feature_columns])
        return df
    
    def prepare_data(self, input_file: str, target_period: int = 12) -> None:
        """准备训练数据
        
        完整的数据预处理流程，包括加载、特征工程、目标变量创建等
        
        Args:
            input_file: 输入文件路径
            target_period: 预测周期
        """
        try:
            # 加载数据
            logger.info(f"加载数据: {input_file}")
            df = self.load_data(input_file)
            
            # 数据质量检查
            logger.info("进行数据质量检查...")
            self._check_data_quality(df)
            
            # 添加技术指标
            logger.info("添加技术指标...")
            df = self.add_technical_indicators(df)
            
            # 创建目标变量
            logger.info("创建目标变量...")
            df = self.create_target(df, target_period)
            
            # 处理缺失值
            logger.info("处理缺失值...")
            df = self.handle_missing_values(df)
            
            # 处理异常值
            logger.info("处理异常值...")
            df = self._handle_outliers(df)
            
            # 标准化特征
            logger.info("标准化特征...")
            df, scaler = self.normalize_features(df, self.feature_columns)
            
            # 保存处理后的数据
            output_file = self.processed_dir / f"processed_{Path(input_file).name}"
            df.to_csv(output_file)
            logger.info(f"处理后的数据已保存到: {output_file}")
            
            # 保存scaler
            self.save_scaler(scaler, self.scaler_file)
            logger.info(f"Scaler已保存到: {self.scaler_file}")
            
            # 生成数据质量报告
            logger.info("生成数据质量报告...")
            self._generate_quality_report(df, output_file)
            
        except Exception as e:
            logger.error(f"数据预处理过程中发生错误: {str(e)}")
            raise
            
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """检查数据质量
        
        检查数据的完整性、一致性和异常值
        
        Args:
            df: 原始数据DataFrame
        """
        # 检查数据完整性
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"发现缺失值:\n{missing_values[missing_values > 0]}")
            
        # 检查数据类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"列 {col} 不是数值类型")
                
        # 检查数据一致性
        if (df['high'] < df['low']).any():
            logger.error("发现最高价低于最低价的异常数据")
            
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            logger.error("发现最高价低于开盘价或收盘价的异常数据")
            
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            logger.error("发现最低价高于开盘价或收盘价的异常数据")
            
        # 检查时间序列完整性
        time_diff = df.index.to_series().diff()
        if time_diff.nunique() > 2:  # 允许第一个值为NaT
            logger.warning("时间序列存在间隔不一致的情况")
            
        # 检查异常值
        for col in ['open', 'high', 'low', 'close', 'volume']:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            if (z_scores > 3).any():
                logger.warning(f"列 {col} 存在超过3个标准差的异常值")
                
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理异常值
        
        使用多种方法处理异常值，包括截断、平滑和替换
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 对价格数据进行处理
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            # 计算移动中位数
            median = df[col].rolling(window=5, center=True).median()
            # 计算移动标准差
            std = df[col].rolling(window=5, center=True).std()
            # 定义异常值界限
            upper_bound = median + 3 * std
            lower_bound = median - 3 * std
            # 将异常值替换为界限值
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
        # 对成交量数据进行处理
        # 使用对数变换处理成交量
        df['volume'] = np.log1p(df['volume'])
        
        return df
        
    def _generate_quality_report(self, df: pd.DataFrame, output_file: Path) -> None:
        """生成数据质量报告
        
        生成包含数据统计信息和质量指标的报告
        
        Args:
            df: 处理后的数据DataFrame
            output_file: 输出文件路径
        """
        report = {
            '数据基本信息': {
                '行数': len(df),
                '列数': len(df.columns),
                '时间范围': f"{df.index.min()} 到 {df.index.max()}",
                '缺失值比例': df.isnull().mean().mean(),
            },
            '特征统计': {
                '价格统计': {
                    '开盘价': df['open'].describe().to_dict(),
                    '最高价': df['high'].describe().to_dict(),
                    '最低价': df['low'].describe().to_dict(),
                    '收盘价': df['close'].describe().to_dict(),
                },
                '成交量统计': {
                    '成交量': df['volume'].describe().to_dict(),
                },
                '技术指标统计': {
                    'RSI': df['rsi'].describe().to_dict(),
                    'MACD': df['macd'].describe().to_dict(),
                    '布林带宽度': df['bb_width'].describe().to_dict(),
                }
            },
            '目标变量统计': {
                '价格变化': df['price_change_pct'].describe().to_dict(),
                '波动率': df['future_volatility'].describe().to_dict(),
                '趋势强度': df['trend_strength'].describe().to_dict(),
            }
        }
        
        # 保存报告
        report_file = output_file.parent / f"quality_report_{output_file.stem}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"数据质量报告已保存到: {report_file}")

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