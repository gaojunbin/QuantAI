from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """数据配置类
    
    包含数据获取、预处理和特征工程的所有相关配置参数
    """
    # 数据获取配置
    symbol: str = 'BTCUSDT'  # 交易对，使用BTC因为流动性最好
    interval: str = '5m'     # K线间隔，5分钟可以平衡噪音和时效性
    start_date: str = '2023-01-01'  # 训练数据开始日期，使用较长时间的数据
    end_date: Optional[str] = '2024-04-01'  # 训练数据结束日期
    
    # 数据预处理配置
    seq_length: int = 48     # 输入序列长度，约4小时的数据
    target_period: int = 12  # 预测未来12个时间单位（1小时）
    train_ratio: float = 0.85  # 训练集比例，保留更多数据用于训练
    batch_size: int = 256    # 批次大小，较小的batch size有助于提高泛化能力
    
    # 特征配置
    feature_columns: List[str] = field(default_factory=lambda: [
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
        'stoch_k', 'stoch_d',           # 随机指标
        
        # 价格变化特征
        'price_change_pct',             # 价格变化百分比
        'future_volatility',            # 未来波动率
        'trend_strength',               # 趋势强度
        'price_momentum',               # 价格动量
        'price_acceleration',           # 价格加速度
        'price_range'                   # 价格波动范围
    ])
    
    # 目标变量配置
    target_columns: List[str] = field(default_factory=lambda: [
        'future_price',        # 预测的未来价格
        'price_change_pct',    # 价格变化百分比
        'future_volatility',   # 未来波动率
        'trend_strength'       # 趋势强度
    ])
    
    # 数据目录配置
    data_dir: Path = Path('data')
    raw_dir: Path = data_dir / 'raw'        # 原始数据目录
    processed_dir: Path = data_dir / 'processed'  # 处理后的数据目录

@dataclass
class ModelConfig:
    """模型配置类
    
    包含模型架构、训练参数和保存路径的配置
    """
    # 模型架构配置
    input_dim: int = 31      # 输入特征维度
    d_model: int = 128       # Transformer模型维度
    nhead: int = 8          # 注意力头数
    num_encoder_layers: int = 4  # 编码器层数
    dim_feedforward: int = 512   # 前馈网络维度
    dropout: float = 0.2     # Dropout比率，防止过拟合
    max_seq_length: int = 100    # 最大序列长度
    
    # 训练配置
    learning_rate: float = 0.0005  # 学习率，较小的学习率有助于稳定训练
    n_epochs: int = 300           # 训练轮数，确保充分学习
    
    # 损失函数权重配置
    price_loss_weight: float = 1.0      # 价格预测损失权重
    volatility_loss_weight: float = 0.5  # 波动率预测损失权重
    trend_loss_weight: float = 0.3       # 趋势预测损失权重
    
    # 模型保存配置
    model_dir: Path = Path('models')
    best_model_path: Path = model_dir / 'best_model.pth'  # 最佳模型保存路径
    evaluation_dir: Path = model_dir / 'evaluation'       # 评估结果保存目录

@dataclass
class TradingConfig:
    """交易配置类
    
    包含交易参数、风险管理和报告生成的配置
    """
    # 交易参数配置
    initial_capital: float = 10000.0  # 初始资金
    position_size: float = 0.05       # 单次交易资金比例
    max_position: float = 0.5         # 最大持仓比例
    stop_loss: float = 0.015          # 止损比例
    take_profit: float = 0.03         # 止盈比例
    
    # 风险管理配置
    max_daily_loss: float = 0.02      # 每日最大亏损限制
    max_drawdown: float = 0.1         # 最大回撤限制
    trailing_stop: float = 0.01       # 追踪止损比例
    
    # 价格预测阈值配置
    min_price_change: float = 0.005   # 最小价格变化阈值，过滤小波动
    min_trend_strength: float = 0.6    # 最小趋势强度阈值，确保趋势明显
    max_volatility: float = 0.03       # 最大波动率阈值，避免高波动期交易
    
    # 交易报告配置
    report_dir: Path = Path('reports')
    equity_curve_path: Path = report_dir / 'equity_curve.png'  # 权益曲线图保存路径
    trading_report_path: Path = report_dir / 'trading_report.csv'  # 交易报告保存路径

@dataclass
class Config:
    """总配置类
    
    整合所有配置，并提供目录初始化功能
    """
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    trading: TradingConfig = TradingConfig()
    
    def __post_init__(self):
        """初始化必要的目录结构"""
        # 创建数据目录
        self.data.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建模型目录
        self.model.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建报告目录
        self.trading.report_dir.mkdir(parents=True, exist_ok=True)

# 创建默认配置实例
config = Config() 