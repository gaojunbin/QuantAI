import logging
from pathlib import Path
import argparse
from datetime import datetime

from src.data.fetch_data import BinanceDataFetcher
from src.data.preprocess import DataPreprocessor
from src.models.transformer import TimeSeriesTransformer
from src.training.train import ModelTrainer, prepare_data
from src.training.evaluate import ModelEvaluator
from src.utils.metrics import plot_equity_curve, generate_trading_report
from configs.config import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI量化交易系统')
    parser.add_argument('--mode', type=str, required=True,
                      choices=['fetch', 'preprocess', 'train', 'evaluate', 'all'],
                      help='运行模式')
    parser.add_argument('--symbol', type=str, default=config.data.symbol,
                      help='交易对符号')
    parser.add_argument('--interval', type=str, default=config.data.interval,
                      help='K线间隔')
    parser.add_argument('--start-date', type=str, default=config.data.start_date,
                      help='开始日期')
    parser.add_argument('--end-date', type=str, default=None,
                      help='结束日期')
    return parser.parse_args()

def fetch_data(args):
    """获取数据"""
    logger.info("开始获取数据...")
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_historical_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_str=args.start_date,
        end_str=args.end_date
    )
    
    # 保存数据
    output_file = config.data.raw_dir / f"{args.symbol}_{args.interval}_{args.start_date}_{args.end_date or datetime.now().strftime('%Y-%m-%d')}.csv"
    df.to_csv(output_file)
    logger.info(f"数据已保存到: {output_file}")

def preprocess_data(args):
    """预处理数据"""
    logger.info("开始预处理数据...")
    preprocessor = DataPreprocessor()
    
    # 处理所有原始数据文件
    raw_files = list(config.data.raw_dir.glob(f"{args.symbol}_{args.interval}_*.csv"))
    
    for file in raw_files:
        logger.info(f"处理文件: {file}")
        preprocessor.prepare_data(str(file))

def train_model(args):
    """训练模型"""
    logger.info("开始训练模型...")
    
    # 准备数据
    processed_files = list(config.data.processed_dir.glob(f"processed_{args.symbol}_{args.interval}_*.csv"))
    if not processed_files:
        raise FileNotFoundError("没有找到处理后的数据文件")
    
    data_path = str(processed_files[0])
    train_loader, val_loader = prepare_data(
        data_path=data_path,
        seq_length=config.data.seq_length,
        batch_size=config.data.batch_size
    )
    
    # 创建模型
    model = TimeSeriesTransformer(
        input_dim=len(config.data.feature_columns),
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_encoder_layers=config.model.num_encoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_seq_length=config.model.max_seq_length
    )
    
    # 训练模型
    trainer = ModelTrainer(model)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.model.n_epochs,
        learning_rate=config.model.learning_rate,
        save_path=config.model.best_model_path
    )

def evaluate_model(args):
    """评估模型"""
    logger.info("开始评估模型...")
    
    # 准备数据
    processed_files = list(config.data.processed_dir.glob(f"processed_{args.symbol}_{args.interval}_*.csv"))
    if not processed_files:
        raise FileNotFoundError("没有找到处理后的数据文件")
    
    data_path = str(processed_files[0])
    _, val_loader = prepare_data(
        data_path=data_path,
        seq_length=config.data.seq_length,
        batch_size=config.data.batch_size
    )
    
    # 创建模型
    model = TimeSeriesTransformer(
        input_dim=len(config.data.feature_columns),
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        num_encoder_layers=config.model.num_encoder_layers,
        dim_feedforward=config.model.dim_feedforward,
        dropout=config.model.dropout,
        max_seq_length=config.model.max_seq_length
    )
    
    # 加载模型权重
    model.load_state_dict(torch.load(config.model.best_model_path))
    
    # 评估模型
    evaluator = ModelEvaluator(model)
    metrics, probs, targets = evaluator.evaluate(val_loader)
    
    # 打印评估指标
    logger.info("评估指标:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # 生成交易报告
    report = generate_trading_report(
        predictions=(probs > 0.5).astype(int),
        actual_returns=val_loader.dataset.targets.numpy(),
        prices=val_loader.dataset.data[:, 3].numpy(),  # 使用收盘价
        save_path=config.trading.trading_report_path
    )
    
    # 绘制权益曲线
    plot_equity_curve(
        returns=val_loader.dataset.targets.numpy(),
        predictions=(probs > 0.5).astype(int),
        save_path=config.trading.equity_curve_path
    )

def main():
    """主函数"""
    args = parse_args()
    
    if args.mode == 'fetch' or args.mode == 'all':
        fetch_data(args)
    
    if args.mode == 'preprocess' or args.mode == 'all':
        preprocess_data(args)
    
    if args.mode == 'train' or args.mode == 'all':
        train_model(args)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        evaluate_model(args)

if __name__ == "__main__":
    main() 