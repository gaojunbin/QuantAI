from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """数据配置"""
    # 数据获取配置
    symbol: str = 'BNBUSDT'
    interval: str = '1m'
    start_date: str = '2024-04-01'
    end_date: Optional[str] = '2025-04-01'
    
    # 数据预处理配置
    seq_length: int = 24
    target_period: int = 24
    train_ratio: float = 0.8
    batch_size: int = 1280
    
    # 特征配置
    feature_columns: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'sma_20', 'sma_50', 'ema_20', 'rsi', 'macd',
        'bb_high', 'bb_low', 'obv'
    ])
    
    # 数据目录
    data_dir: Path = Path('data')
    raw_dir: Path = data_dir / 'raw'
    processed_dir: Path = data_dir / 'processed'

@dataclass
class ModelConfig:
    """模型配置"""
    # 模型架构配置
    input_dim: int = 13
    d_model: int = 64
    nhead: int = 8
    num_encoder_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_length: int = 100
    
    # 训练配置
    learning_rate: float = 0.001
    n_epochs: int = 200
    
    # 模型保存配置
    model_dir: Path = Path('models')
    best_model_path: Path = model_dir / 'best_model.pth'
    evaluation_dir: Path = model_dir / 'evaluation'

@dataclass
class TradingConfig:
    """交易配置"""
    # 交易参数
    initial_capital: float = 10000.0
    position_size: float = 0.1  # 每次交易使用资金比例
    max_position: float = 1.0   # 最大持仓比例
    stop_loss: float = 0.02     # 止损比例
    take_profit: float = 0.05   # 止盈比例
    
    # 交易报告配置
    report_dir: Path = Path('reports')
    equity_curve_path: Path = report_dir / 'equity_curve.png'
    trading_report_path: Path = report_dir / 'trading_report.csv'

@dataclass
class Config:
    """总配置"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    trading: TradingConfig = TradingConfig()
    
    def __post_init__(self):
        """创建必要的目录"""
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