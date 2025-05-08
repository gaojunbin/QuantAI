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
                 d_model: int = 64,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_length: int = 100):
        """
        初始化时间序列Transformer模型
        
        Args:
            input_dim: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_encoder_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_seq_length: 最大序列长度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
            mask: 注意力掩码，形状为 [seq_len, seq_len]
            
        Returns:
            输出张量，形状为 [batch_size, 1]
        """

        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        # 位置编码
        x = self.pos_encoder(x)
        
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        
        # Transformer编码
        x = self.transformer_encoder(x, mask)

        
        # 只使用最后一个时间步的输出
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # 输出层
        x = self.output_layer(x)  # [batch_size, 1]
        
        return x

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