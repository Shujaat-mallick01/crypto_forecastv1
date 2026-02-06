"""
Visualization module for predictions and model analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 150


class PredictionVisualizer:
    """Visualization tools for model predictions and analysis."""
    
    def __init__(self, save_path: Optional[Path] = None):
        self.save_path = Path(save_path) if save_path else None
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
    
    def plot_actual_vs_predicted(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Actual vs Predicted",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot actual vs predicted values."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1 = axes[0]
        x = dates if dates is not None else range(len(y_true))
        ax1.plot(x, y_true, label='Actual', color='blue', linewidth=2)
        ax1.plot(x, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
        ax1.set_xlabel('Date' if dates is not None else 'Index')
        ax1.set_ylabel('Value')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.scatter(y_true, y_pred, alpha=0.5)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Prediction Scatter Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_name and self.save_path:
            plt.savefig(self.save_path / f"{save_name}.png", bbox_inches='tight')
        return fig
    
    def plot_forecast_timeline(
        self, current_value: float, predictions: Dict[int, float],
        current_date: pd.Timestamp, symbol: str, target: str = 'price',
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot forecast timeline."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dates, values = [current_date], [current_value]
        for horizon, pred in sorted(predictions.items()):
            dates.append(current_date + pd.Timedelta(days=horizon))
            values.append(pred)
        
        ax.axhline(y=current_value, color='blue', linestyle='--', 
                   label=f'Current: ${current_value:,.2f}', linewidth=2)
        ax.plot(dates, values, 'o-', color='red', linewidth=2, markersize=10, label='Predictions')
        
        for date, value in zip(dates[1:], values[1:]):
            change = (value - current_value) / current_value * 100
            ax.annotate(f'{change:+.1f}%', xy=(date, value), xytext=(0, 15),
                       textcoords='offset points', ha='center', fontsize=10,
                       color='green' if change > 0 else 'red', weight='bold')
        
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{target.title()} (USD)')
        ax.set_title(f'{symbol} - {target.title()} Forecast', weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        if save_name and self.save_path:
            plt.savefig(self.save_path / f"{save_name}.png", bbox_inches='tight')
        return fig
    
    def plot_feature_importance(
        self, importance_df: pd.DataFrame, top_n: int = 20,
        title: str = "Feature Importance", save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature importance."""
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.nlargest(top_n, 'importance')
        
        ax.barh(range(len(top_features)), top_features['importance'],
               color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features))))
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, weight='bold')
        
        plt.tight_layout()
        if save_name and self.save_path:
            plt.savefig(self.save_path / f"{save_name}.png", bbox_inches='tight')
        return fig
    
    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        title: str = "Residual Analysis", save_name: Optional[str] = None
    ) -> plt.Figure:
        """Plot residual analysis."""
        residuals = y_true - y_pred
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_title('Residual Distribution')
        
        axes[1, 0].plot(residuals, alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Index')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Over Time')
        
        axes[1, 1].scatter(range(len(residuals)), np.abs(residuals), alpha=0.5)
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Absolute Residuals')
        axes[1, 1].set_title('Absolute Residuals')
        
        plt.suptitle(title, fontsize=14, weight='bold')
        plt.tight_layout()
        if save_name and self.save_path:
            plt.savefig(self.save_path / f"{save_name}.png", bbox_inches='tight')
        return fig
