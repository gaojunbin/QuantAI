import os
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
from dotenv import load_dotenv
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class BinanceDataFetcher:
    def __init__(self):
        """初始化币安数据获取器"""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.client = Client(self.api_key, self.api_secret)
        
    def fetch_historical_klines(self, symbol: str, interval: str, 
                              start_str: str, end_str: str = None) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号，例如 'BTCUSDT'
            interval: K线间隔，例如 '1h', '4h', '1d'
            start_str: 开始时间，格式：'YYYY-MM-DD'
            end_str: 结束时间，格式：'YYYY-MM-DD'，默认为当前时间
            
        Returns:
            DataFrame: 包含K线数据的DataFrame
        """
        try:
            # 转换时间格式
            start_time = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            end_time = int(datetime.strptime(end_str, '%Y-%m-%d').timestamp() * 1000) if end_str else None
            
            # 获取K线数据
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据处理
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 转换数据类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                             'quote_asset_volume', 'taker_buy_base_asset_volume',
                             'taker_buy_quote_asset_volume']
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            
            # 删除不需要的列
            df.drop(['close_time', 'ignore'], axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"获取数据时发生错误: {str(e)}")
            raise

def main():
    """主函数"""
    # 创建数据目录
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据获取器
    fetcher = BinanceDataFetcher()
    
    # 获取BTC/USDT的1小时K线数据
    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '2023-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        logger.info(f"开始获取 {symbol} 的历史数据...")
        df = fetcher.fetch_historical_klines(symbol, interval, start_date, end_date)
        
        # 保存数据
        output_file = data_dir / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        df.to_csv(output_file)
        logger.info(f"数据已保存到: {output_file}")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 