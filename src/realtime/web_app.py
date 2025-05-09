from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List
import threading
import time
import os

from src.realtime.predictor import RealtimePredictor
from src.realtime.data_collector import BinanceDataCollector
from configs.config import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# 全局变量
predictor = None
data_collector = None
prediction_thread = None
stop_prediction = False

def create_price_chart(symbol: str, prediction_time: datetime, current_price: float) -> str:
    """
    创建价格图表
    
    Args:
        symbol: 交易对符号
        prediction_time: 预测时间
        current_price: 当前价格
        
    Returns:
        图表的HTML字符串
    """
    # 获取历史数据
    df = data_collector.get_historical_klines(
        symbol=symbol,
        interval=config.data.interval,
        lookback_periods=config.data.seq_length
    )
    
    # 创建图表
    fig = go.Figure()
    
    # 添加价格线
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        mode='lines',
        name='价格'
    ))
    
    # 添加预测点
    fig.add_trace(go.Scatter(
        x=[prediction_time],
        y=[current_price],
        mode='markers',
        marker=dict(
            size=10,
            color='red'
        ),
        name='预测点'
    ))
    
    # 设置布局
    fig.update_layout(
        title=f'{symbol} 价格走势',
        xaxis_title='时间',
        yaxis_title='价格',
        template='plotly_white'
    )
    
    return fig.to_html(full_html=False)


def prediction_worker():
    """
    预测工作线程
    """
    global stop_prediction
    
    while not stop_prediction:
        try:
            # 进行预测
            result = predictor.make_prediction(
                config.data.symbol,
                config.data.interval
            )
            
            # 等待一段时间再进行下一次预测
            time.sleep(60)  # 每分钟进行一次预测
            
        except ValueError as e:
            logger.error(f"数据验证失败: {str(e)}")
            time.sleep(300)  # 数据验证失败时等待5分钟再重试
        except Exception as e:
            logger.error(f"预测工作线程发生错误: {str(e)}")
            time.sleep(60)  # 其他错误时等待1分钟再重试

@app.route('/')
def index():
    """
    主页
    """
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """
    获取预测列表
    """
    predictions = []
    for file_path in Path('data/predictions').glob('*.json'):
        with open(file_path, 'r') as f:
            predictions.append(json.load(f))
    
    # 按时间排序
    predictions.sort(key=lambda x: x['prediction_time'], reverse=True)
    
    return jsonify(predictions)

@app.route('/api/prediction/<prediction_id>')
def get_prediction(prediction_id):
    """
    获取单个预测详情
    """
    try:
        result = predictor._load_prediction(prediction_id)
        
        # 创建价格图表
        prediction_time = datetime.fromisoformat(result['prediction_time'])
        chart_html = create_price_chart(
            result['symbol'],
            prediction_time,
            result['current_price']
        )
        
        result['chart_html'] = chart_html
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/stats')
def get_stats():
    """
    获取预测统计信息
    """
    return jsonify(predictor.get_prediction_stats())

@app.route('/api/start_prediction', methods=['POST'])
def start_prediction():
    """
    开始预测
    """
    global prediction_thread, stop_prediction
    
    if prediction_thread is None or not prediction_thread.is_alive():
        stop_prediction = False
        prediction_thread = threading.Thread(target=prediction_worker)
        prediction_thread.start()
        return jsonify({'status': 'started'})
    
    return jsonify({'status': 'already_running'})

@app.route('/api/stop_prediction', methods=['POST'])
def stop_prediction_route():
    """
    停止预测
    """
    global stop_prediction
    
    stop_prediction = True
    if prediction_thread is not None:
        prediction_thread.join()
    
    return jsonify({'status': 'stopped'})

def init_app(model_path: str):
    """
    初始化应用
    
    Args:
        model_path: 模型权重文件路径
    """
    global predictor, data_collector
    
    # 初始化预测器和数据收集器
    predictor = RealtimePredictor(
        model_path,
        seq_length=config.data.seq_length,
        prediction_horizon=config.data.target_period
    )
    data_collector = BinanceDataCollector()
    
    # 创建模板目录
    templates_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    templates_dir.mkdir(exist_ok=True)
    
    # 创建主页模板
    template_path = templates_dir / 'index.html'
    if not template_path.exists():
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Quant AI 价格趋势预测系统</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .prediction-card {
            margin-bottom: 20px;
        }
        .stats-card {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="mb-4">Quant AI 价格趋势预测系统</h1>
        
        <!-- 控制按钮 -->
        <div class="row mb-4">
            <div class="col">
                <button id="startBtn" class="btn btn-primary">开始预测</button>
                <button id="stopBtn" class="btn btn-danger">停止预测</button>
            </div>
        </div>
        
        <!-- 统计信息 -->
        <div class="row mb-4">
            <div class="col">
                <div class="card stats-card">
                    <div class="card-header">
                        预测统计
                    </div>
                    <div class="card-body">
                        <div id="stats"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 预测列表 -->
        <div class="row">
            <div class="col">
                <h2>预测历史</h2>
                <div id="predictions"></div>
            </div>
        </div>
    </div>
    
    <script>
        // 更新统计信息
        function updateStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => {
                    const statsHtml = `
                        <p>总预测数: ${data.total_predictions}</p>
                        <p>已验证预测: ${data.verified_predictions}</p>
                        <p>准确率: ${(data.accuracy * 100).toFixed(2)}%</p>
                        <p>正确预测: ${data.correct_predictions}</p>
                        <p>错误预测: ${data.incorrect_predictions}</p>
                    `;
                    document.getElementById('stats').innerHTML = statsHtml;
                });
        }
        
        // 更新预测列表
        function updatePredictions() {
            fetch('/api/predictions')
                .then(response => response.json())
                .then(predictions => {
                    const predictionsHtml = predictions.map(prediction => `
                        <div class="card prediction-card">
                            <div class="card-header">
                                ${prediction.symbol} - ${new Date(prediction.prediction_time).toLocaleString()}
                            </div>
                            <div class="card-body">
                                <p>当前价格: ${prediction.current_price}</p>
                                <p>预测方向: ${prediction.prediction === 1 ? '上涨' : '下跌'}</p>
                                <p>置信度: ${(prediction.confidence * 100).toFixed(2)}%</p>
                                ${prediction.verified ? `
                                    <p>实际方向: ${prediction.actual_direction === 1 ? '上涨' : '下跌'}</p>
                                    <p>预测结果: ${prediction.is_correct ? '正确' : '错误'}</p>
                                ` : '<p>等待验证...</p>'}
                                <div id="chart-${prediction.symbol}_${prediction.prediction_time}"></div>
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('predictions').innerHTML = predictionsHtml;
                    
                    // 加载每个预测的图表
                    predictions.forEach(prediction => {
                        const predictionId = `${prediction.symbol}_${prediction.prediction_time}`;
                        fetch(`/api/prediction/${predictionId}`)
                            .then(response => response.json())
                            .then(data => {
                                const chartDiv = document.getElementById(`chart-${predictionId}`);
                                if (chartDiv) {
                                    chartDiv.innerHTML = data.chart_html;
                                }
                            });
                    });
                });
        }
        
        // 开始预测
        document.getElementById('startBtn').addEventListener('click', () => {
            fetch('/api/start_prediction', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started') {
                        alert('预测已开始');
                    }
                });
        });
        
        // 停止预测
        document.getElementById('stopBtn').addEventListener('click', () => {
            fetch('/api/stop_prediction', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'stopped') {
                        alert('预测已停止');
                    }
                });
        });
        
        // 定期更新数据
        setInterval(() => {
            updateStats();
            updatePredictions();
        }, 60000);  // 每分钟更新一次
        
        // 初始加载
        updateStats();
        updatePredictions();
    </script>
</body>
</html>
            ''')
    
    return app

if __name__ == '__main__':
    app = init_app('models/best_model.pth')
    app.run(debug=True, host='0.0.0.0', port=5000) 