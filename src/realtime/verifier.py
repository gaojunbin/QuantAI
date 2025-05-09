import json
from pathlib import Path
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List
import threading

from src.realtime.predictor import RealtimePredictor
from configs.config import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionVerifier:
    def __init__(self, model_path: str):
        """
        初始化预测验证器
        
        Args:
            model_path: 模型权重文件路径
        """
        self.predictor = RealtimePredictor(
            model_path,
            seq_length=config.data.seq_length,
            prediction_horizon=config.data.target_period
        )
        self.predictions_dir = Path('data/predictions')
        self.stop_verification = False
        
    def get_wait_minutes(self, interval: str) -> int:
        """
        根据时间间隔计算分钟数
        
        Args:
            interval: 时间间隔字符串 (例如: '1m', '5m', '1h', '4h', '1d')
            
        Returns:
            对应的分钟数
        """
        unit = interval[-1]
        value = int(interval[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 24 * 60
        else:
            raise ValueError(f"不支持的时间间隔单位: {unit}")
    
    def get_pending_predictions(self) -> List[Dict]:
        """
        获取待验证的预测列表
        
        Returns:
            待验证的预测列表
        """
        pending_predictions = []
        
        for file_path in self.predictions_dir.glob('*.json'):
            try:
                with open(file_path, 'r') as f:
                    prediction = json.load(f)
                    
                # 检查是否已验证
                if prediction.get('verified', False):
                    continue
                    
                # 计算预测时间
                prediction_time = datetime.fromisoformat(prediction['prediction_time'])
                wait_minutes = self.get_wait_minutes(config.data.interval) * self.predictor.prediction_horizon
                verification_time = prediction_time + timedelta(minutes=wait_minutes)
                
                # 如果已经到达验证时间，添加到待验证列表
                if datetime.now() >= verification_time:
                    pending_predictions.append({
                        'file_path': file_path,
                        'prediction': prediction,
                        'prediction_id': f"{prediction['symbol']}_{prediction['prediction_time']}"
                    })
                    
            except Exception as e:
                logger.error(f"读取预测文件 {file_path} 时发生错误: {str(e)}")
                continue
                
        return pending_predictions
    
    def verify_predictions(self):
        """
        验证预测结果
        """
        while not self.stop_verification:
            try:
                # 获取待验证的预测
                pending_predictions = self.get_pending_predictions()
                
                # 验证每个预测
                for pred in pending_predictions:
                    try:
                        logger.info(f"验证预测: {pred['prediction_id']}")
                        self.predictor.verify_prediction(pred['prediction_id'])
                    except Exception as e:
                        logger.error(f"验证预测 {pred['prediction_id']} 时发生错误: {str(e)}")
                
                # 等待一段时间再检查
                time.sleep(30)  # 每30秒检查一次
                
            except Exception as e:
                logger.error(f"验证过程中发生错误: {str(e)}")
                time.sleep(60)  # 发生错误时等待1分钟再重试
    
    def start(self):
        """启动验证器"""
        self.stop_verification = False
        self.verification_thread = threading.Thread(target=self.verify_predictions)
        self.verification_thread.start()
        logger.info("预测验证器已启动")
    
    def stop(self):
        """停止验证器"""
        self.stop_verification = True
        if hasattr(self, 'verification_thread'):
            self.verification_thread.join()
        logger.info("预测验证器已停止")

def main():
    """主函数"""
    verifier = PredictionVerifier('models/best_model.pth')
    
    try:
        verifier.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        verifier.stop()

if __name__ == "__main__":
    main() 