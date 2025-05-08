import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta     # pip install python-dateutil
from dotenv import load_dotenv
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量（保留，不再强制需要 Key）
load_dotenv()

class BinanceDataFetcher:
    """
    从 https://data.binance.vision 下载压缩包，不依赖 Binance API Key。
    """
    BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

    def __init__(self):
        """初始化——不再连接 Binance API，仅保留占位符以保证向后兼容。"""
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        # 如果后续代码仍想用 self.client，可给个占位对象
        self.client = None

    # ---------- 私有辅助函数 ----------
    def _months_between(self, start: datetime, end: datetime):
        cur = datetime(start.year, start.month, 1)
        end = datetime(end.year, end.month, 1)
        while cur <= end:
            yield cur
            cur += relativedelta(months=1)

    def _build_url(self, symbol: str, interval: str, dt: datetime) -> str:
        return (f"{self.BASE_URL}/{symbol}/{interval}/"
                f"{symbol}-{interval}-{dt.year}-{dt.month:02d}.zip")
        
    # ---------- 对外主接口 ----------
    def fetch_historical_klines(self, symbol: str, interval: str,
                                start_str: str, end_str: str = None) -> pd.DataFrame:
        """
        数据来源为下载 ZIP。
        """
        try:
            symbol = symbol.upper()
            interval = interval.lower()

            start_date = datetime.strptime(start_str, '%Y-%m-%d')
            end_date = (datetime.strptime(end_str, '%Y-%m-%d')
                        if end_str else datetime.utcnow())

            # 按月循环下载
            frames = []
            for month_dt in self._months_between(start_date, end_date):
                url = self._build_url(symbol, interval, month_dt)
                logger.info(f"⇣ downloading {url}")
                resp = requests.get(url, stream=True)
                if resp.status_code == 404:
                    logger.warning(f"⚠️  file not found, skip: {url}")
                    continue
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df_month = pd.read_csv(
                            f,
                            header=None,
                            names=[
                                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                'close_time', 'quote_asset_volume', 'number_of_trades',
                                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                                'ignore'
                            ],
                            dtype={'timestamp': 'Int64'},      # 明确要求整数
                            engine='python',                   # 允许跳过坏行
                            on_bad_lines='skip'                # pandas ≥1.3
                        )
                        # 先把非数字、溢出的时间戳剔掉
                        df_month['timestamp'] = pd.to_numeric(df_month['timestamp'],
                                              errors='coerce')
                        df_month.dropna(subset=['timestamp'], inplace=True)
                        # 转换时容错，溢出变 NaT，再统一删掉
                        df_month['timestamp'] = pd.to_datetime(df_month['timestamp'],
                                                            unit='ms',
                                                            errors='coerce')
                        
                        df_month.dropna(subset=['timestamp'], inplace=True)
                        frames.append(df_month)

            if not frames:
                raise RuntimeError("未能下载到任何数据，请检查交易对、间隔或日期范围。")

            df = pd.concat(frames, ignore_index=True)

            # 基础清洗
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            numeric_cols = [
                'open', 'high', 'low', 'close', 'volume',
                'quote_asset_volume', 'taker_buy_base_asset_volume',
                'taker_buy_quote_asset_volume'
            ]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df.drop(['close_time', 'ignore'], axis=1, inplace=True)

            # 仅保留用户指定的日期区间
            df = df.loc[start_date:end_date]

            return df

        except Exception as e:
            logger.error(f"获取数据时发生错误: {str(e)}")
            raise


def main():
    data_dir = Path('data/raw')
    data_dir.mkdir(parents=True, exist_ok=True)

    fetcher = BinanceDataFetcher()

    symbol = 'BTCUSDT'
    interval = '1h'
    start_date = '2024-04-30'
    # end_date = datetime.now().strftime('%Y-%m-%d')
    end_date = '2025-04-30'

    try:
        logger.info(f"开始获取 {symbol} 的历史数据...")
        df = fetcher.fetch_historical_klines(symbol, interval, start_date, end_date)

        output_file = data_dir / f"{symbol}_{interval}_{start_date}_{end_date}.csv"
        df.to_csv(output_file)
        logger.info(f"数据已保存到: {output_file}")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()