import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """
    计算收益率
    
    Args:
        prices: 价格序列
        
    Returns:
        收益率序列
    """
    return np.diff(prices) / prices[:-1]

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率
        
    Returns:
        夏普比率
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 年化

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    计算最大回撤
    
    Args:
        returns: 收益率序列
        
    Returns:
        最大回撤
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (running_max - cumulative_returns) / running_max
    return np.max(drawdown)

def calculate_win_rate(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """
    计算胜率
    
    Args:
        predictions: 预测方向（1: 上涨, 0: 下跌）
        actual_returns: 实际收益率
        
    Returns:
        胜率
    """
    predictions = np.asarray(predictions).ravel()
    actual_returns = np.asarray(actual_returns).ravel()
    correct_predictions = np.sum((predictions == 1) & (actual_returns > 0) | 
                               (predictions == 0) & (actual_returns <= 0))
    return correct_predictions / len(predictions)

def calculate_profit_factor(predictions: np.ndarray, actual_returns: np.ndarray) -> float:
    """
    计算盈亏比
    
    Args:
        predictions: 预测方向（1: 上涨, 0: 下跌）
        actual_returns: 实际收益率
        
    Returns:
        盈亏比
    """
    predictions = np.asarray(predictions).ravel()
    actual_returns = np.asarray(actual_returns).ravel()
    # 计算盈利和亏损
    profits = actual_returns[(predictions == 1) & (actual_returns > 0)]
    losses = actual_returns[(predictions == 1) & (actual_returns <= 0)]
    
    total_profit = np.sum(profits)
    total_loss = abs(np.sum(losses))
    
    return total_profit / total_loss if total_loss != 0 else float('inf')

def calculate_trading_metrics(predictions: np.ndarray,
                            actual_returns: np.ndarray,
                            prices: np.ndarray) -> Dict[str, float]:
    """
    计算交易指标
    
    Args:
        predictions: 预测方向（1: 上涨, 0: 下跌）
        actual_returns: 实际收益率
        prices: 价格序列
        
    Returns:
        包含各项指标的字典
    """
    # 计算策略收益率
    strategy_returns = actual_returns * (2 * predictions - 1)
    
    # 计算累积收益率
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # 计算各项指标
    metrics = {
        'total_return': cumulative_returns[-1] - 1,
        'annual_return': (cumulative_returns[-1] ** (252 / len(strategy_returns)) - 1),
        'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'win_rate': calculate_win_rate(predictions, actual_returns),
        'profit_factor': calculate_profit_factor(predictions, actual_returns)
    }
    
    return metrics

def plot_equity_curve(returns: np.ndarray,
                     predictions: np.ndarray = None,
                     title: str = 'Equity Curve',
                     save_path: str = None):
    """
    绘制权益曲线
    
    Args:
        returns: 收益率序列
        predictions: 预测方向（可选）
        title: 图表标题
        save_path: 图像保存路径
    """
    import matplotlib.pyplot as plt

    returns = np.asarray(returns, dtype=np.float64).ravel()
    if predictions is not None:
        predictions = np.asarray(predictions).ravel()
    
    # 计算累积收益率（使用对数累积避免浮点溢出）
    if predictions is not None:
        strategy_returns = returns * (2 * predictions - 1)
    else:
        strategy_returns = returns
    
    # 使用 log1p / exp 累积，处理小于等于 -1 的极端值，避免 -inf
    strategy_returns_safe = np.clip(strategy_returns, -0.999999, None)
    returns_safe = np.clip(returns, -0.999999, None)
    cumulative_returns = np.exp(np.log1p(strategy_returns_safe).cumsum())
    cumulative_buy_hold = np.exp(np.log1p(returns_safe).cumsum())
    
    # 绘制权益曲线
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns, label='Strategy')
    plt.plot(cumulative_buy_hold, label='Buy & Hold', alpha=0.5)
    plt.title(title)
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_trading_report(predictions: np.ndarray,
                          actual_returns: np.ndarray,
                          prices: np.ndarray,
                          save_path: str = None) -> pd.DataFrame:
    """
    生成交易报告
    
    Args:
        predictions: 预测方向（1: 上涨, 0: 下跌）
        actual_returns: 实际收益率
        prices: 价格序列
        save_path: 报告保存路径
        
    Returns:
        包含交易统计的DataFrame
    """
    # 计算交易指标
    metrics = calculate_trading_metrics(predictions, actual_returns, prices)
    
    # 创建报告DataFrame
    report = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    # 格式化数值
    report['Value'] = report['Value'].apply(lambda x: f'{x:.4f}')
    
    if save_path:
        report.to_csv(save_path, index=False)
    
    return report 