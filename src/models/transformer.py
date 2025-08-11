import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 创建位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.2,
                 max_seq_length: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征提取层
        self.feature_extraction = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # 多头注意力层
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 价格预测层
        self.price_prediction = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 预测具体价格
        )
        
        # 波动率预测层
        self.volatility_prediction = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Softplus()  # 确保波动率为正
        )
        
        # 趋势强度预测层
        self.trend_prediction = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出0-1之间的趋势强度
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> dict:
        # 特征提取
        x = self.feature_extraction(x)
        
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        # Transformer编码
        x = self.transformer_encoder(x, mask)
        
        # 多头注意力
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + attn_output
        
        # 只使用最后一个时间步的输出
        x = x[:, -1, :]
        
        # 多任务预测
        price = self.price_prediction(x)
        volatility = self.volatility_prediction(x)
        trend = self.trend_prediction(x)
        
        return {
            'price': price,
            'volatility': volatility,
            'trend': trend
        }

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data: torch.Tensor,
                 targets: torch.Tensor,
                 seq_length: int):
        """
        初始化时间序列数据集
        
        Args:
            data: 特征数据，形状为 [n_samples, n_features]
            targets: 目标变量，形状为 [n_samples]
            seq_length: 序列长度
        """
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx: int) -> tuple:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (sequence, target) 元组
        """
        sequence = self.data[idx:idx + self.seq_length]
        target = self.targets[idx + self.seq_length - 1]
        return sequence, target 