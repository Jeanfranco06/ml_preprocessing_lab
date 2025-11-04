"""
Model Comparer
Compares multiple trained models and provides insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..config import get_config
from ..utils import get_logger

logger = get_logger(__name__)

class ModelComparer:
    """
    Compares multiple trained models and provides insights.
    """

    def __init__(self, evaluation_results: Dict[str, Dict[str, Any]] = None):
        """
        Initialize model comparer.

        Args:
            evaluation_results: Dictionary of model evaluation results
        """
        self.evaluation_results = evaluation_results or {}
        self.comparison_df = None

    def update_results(self, evaluation_results: Dict[str, Dict[str, Any]]):
        """
        Update evaluation results.

        Args:
            evaluation_results: New evaluation results
        """
        self.evaluation_results = evaluation_results
        self.comparison_df = None  # Reset comparison dataframe

    def create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive comparison DataFrame.

        Returns:
            Comparison DataFrame with all metrics
        """
        if self.comparison_df is not None:
            return self.comparison_df

        comparison_data = []

        for model_name, result in self.evaluation_results.items():
            if 'error' in result:
                row = {
                    'Model': model_name,
                    'Status': 'Error',
                    'Error': result['error']
                }
                # Fill other columns with None
                row.update({col: None for col in self._get_all_metrics()})
                comparison_data.append(row)
                continue

            row = {
                'Model': model_name,
                'Status': 'Success',
                'Problem_Type': result.get('problem_type', 'Unknown'),
                'Error': None
            }

            # Add metrics
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    row[metric] = value

            # Add CV scores if available (from training)
            if 'cv_scores' in result:
                for metric, cv_data in result['cv_scores'].items():
                    if isinstance(cv_data, dict) and 'mean' in cv_data:
                        row[f'{metric}_cv_mean'] = cv_data['mean']
                        row[f'{metric}_cv_std'] = cv_data['std']

            comparison_data.append(row)

        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df

    def _get_all_metrics(self) -> List[str]:
        """
        Get all unique metrics across all models.

        Returns:
            List of metric names
        """
        all_metrics = set()
        for result in self.evaluation_results.values():
            if 'metrics' in result:
                all_metrics.update(result['metrics'].keys())
        return sorted(list(all_metrics))

    def get_best_model(self, metric: str, maximize: bool = True) -> str:
        """
        Get the best performing model for a specific metric.

        Args:
            metric: Metric to optimize
            maximize: Whether to maximize (True) or minimize (False) the metric

        Returns:
            Name of the best model
        """
        df = self.create_comparison_dataframe()

        # Filter out error models and models without the metric
        valid_df = df[(df['Status'] == 'Success') & (df[metric].notna())]

        if valid_df.empty:
            return None

        if maximize:
            best_idx = valid_df[metric].idxmax()
        else:
            best_idx = valid_df[metric].idxmin()

        return valid_df.loc[best_idx, 'Model']

    def get_ranking(self, metric: str, maximize: bool = True) -> pd.DataFrame:
        """
        Get ranking of models for a specific metric.

        Args:
            metric: Metric to rank by
            maximize: Whether to maximize (True) or minimize (False) the metric

        Returns:
            Ranked DataFrame
        """
        df = self.create_comparison_dataframe()

        # Filter out error models and models without the metric
        valid_df = df[(df['Status'] == 'Success') & (df[metric].notna())].copy()

        if valid_df.empty:
            return pd.DataFrame()

        # Sort by metric
        ascending = not maximize
        valid_df = valid_df.sort_values(metric, ascending=ascending)
        valid_df['Rank'] = range(1, len(valid_df) + 1)

        return valid_df[['Rank', 'Model', metric]]

    def plot_metric_comparison(self, metrics: List[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot comparison of models across multiple metrics.

        Args:
            metrics: List of metrics to plot, or None for all available
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        df = self.create_comparison_dataframe()

        # Filter successful models
        success_df = df[df['Status'] == 'Success'].copy()

        if success_df.empty:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No successful model evaluations available',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Get metrics to plot
        if metrics is None:
            metrics = self._get_all_metrics()

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in success_df.columns and success_df[m].notna().any()]

        if not available_metrics:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No metrics available for plotting',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Plot each metric
        for i, metric in enumerate(available_metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Get data for this metric
            metric_data = success_df[['Model', metric]].dropna()

            if metric_data.empty:
                ax.text(0.5, 0.5, f'No data for {metric}',
                       ha='center', va='center', transform=ax.transAxes)
                continue

            # Create bar plot
            bars = ax.bar(range(len(metric_data)), metric_data[metric],
                         color=plt.cm.tab10(np.arange(len(metric_data)) % 10))

            # Customize plot
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(range(len(metric_data)))
            ax.set_xticklabels(metric_data['Model'], rotation=45, ha='right')

            # Add value labels on bars
            for bar, value in zip(bars, metric_data[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       '.3f' if isinstance(value, float) else str(value),
                       ha='center', va='bottom')

        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        plt.tight_layout()
        return fig

    def plot_radar_chart(self, metrics: List[str] = None,
                        figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Create a radar chart comparing models.

        Args:
            metrics: List of metrics to include
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        df = self.create_comparison_dataframe()

        # Filter successful models
        success_df = df[df['Status'] == 'Success'].copy()

        if success_df.empty or len(success_df) < 1:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Need at least 1 successful model for radar chart',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Get metrics to plot
        if metrics is None:
            metrics = self._get_all_metrics()

        # Filter to available numeric metrics
        available_metrics = []
        for metric in metrics:
            if (metric in success_df.columns and
                success_df[metric].notna().any() and
                pd.api.types.is_numeric_dtype(success_df[metric])):
                available_metrics.append(metric)

        if len(available_metrics) < 3:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Need at least 3 numeric metrics for radar chart',
                   ha='center', va='center', transform=ax.transAxes)
            return fig

        # Normalize metrics to 0-1 scale for radar chart
        radar_df = success_df[['Model'] + available_metrics].copy()

        for metric in available_metrics:
            min_val = radar_df[metric].min()
            max_val = radar_df[metric].max()
            if max_val > min_val:
                radar_df[metric] = (radar_df[metric] - min_val) / (max_val - min_val)

        # Create radar chart
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, polar=True)

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, len(available_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot

        # Plot each model
        colors = plt.cm.tab10(np.arange(len(radar_df)) % 10)

        for i, (_, model_data) in enumerate(radar_df.iterrows()):
            values = model_data[available_metrics].tolist()
            values += values[:1]  # Close the plot

            ax.plot(angles, values, 'o-', linewidth=2, label=model_data['Model'],
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Model Comparison Radar Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

        return fig

    def generate_comparison_report(self) -> str:
        """
        Generate a comprehensive comparison report.

        Returns:
            Markdown report string
        """
        df = self.create_comparison_dataframe()

        report = "# Model Comparison Report\n\n"

        # Summary
        total_models = len(df)
        successful_models = len(df[df['Status'] == 'Success'])
        error_models = len(df[df['Status'] == 'Error'])

        report += f"## Summary\n\n"
        report += f"- **Total Models:** {total_models}\n"
        report += f"- **Successful:** {successful_models}\n"
        report += f"- **Errors:** {error_models}\n\n"

        if successful_models == 0:
            report += "No successful model evaluations available.\n"
            return report

        # Best models for each metric
        report += "## Best Models by Metric\n\n"
        report += "| Metric | Best Model | Value |\n"
        report += "|--------|------------|-------|\n"

        metrics = self._get_all_metrics()
        for metric in metrics:
            best_model = self.get_best_model(metric)
            if best_model:
                value = df[df['Model'] == best_model][metric].iloc[0]
                if isinstance(value, float):
                    value_str = ".4f"
                else:
                    value_str = str(value)
                report += f"| {metric} | {best_model} | {value_str} |\n"

        report += "\n"

        # Rankings
        if len(df[df['Status'] == 'Success']) > 1:
            report += "## Rankings\n\n"

            for metric in metrics[:3]:  # Show top 3 metrics
                ranking = self.get_ranking(metric)
                if not ranking.empty:
                    report += f"### {metric.replace('_', ' ').title()} Ranking\n\n"
                    report += "| Rank | Model | Value |\n"
                    report += "|------|-------|-------|\n"

                    for _, row in ranking.iterrows():
                        value = row[metric]
                        if isinstance(value, float):
                            value_str = ".4f"
                        else:
                            value_str = str(value)
                        report += f"| {int(row['Rank'])} | {row['Model']} | {value_str} |\n"

                    report += "\n"

        # Detailed comparison table
        report += "## Detailed Comparison\n\n"
        success_df = df[df['Status'] == 'Success'].drop(['Status', 'Error'], axis=1, errors='ignore')

        if not success_df.empty:
            # Convert to markdown table
            report += success_df.to_markdown(index=False)
            report += "\n\n"

        return report

    def get_model_insights(self) -> Dict[str, Any]:
        """
        Generate insights about the model comparison.

        Returns:
            Dictionary with insights
        """
        df = self.create_comparison_dataframe()
        success_df = df[df['Status'] == 'Success']

        insights = {
            'total_models': len(df),
            'successful_models': len(success_df),
            'error_models': len(df) - len(success_df),
            'best_models': {},
            'consistency_analysis': {},
            'recommendations': []
        }

        if len(success_df) == 0:
            return insights

        # Find best models for each metric
        metrics = self._get_all_metrics()
        for metric in metrics:
            best_model = self.get_best_model(metric)
            if best_model:
                value = success_df[success_df['Model'] == best_model][metric].iloc[0]
                insights['best_models'][metric] = {
                    'model': best_model,
                    'value': value
                }

        # Consistency analysis
        if len(metrics) > 1 and len(success_df) > 1:
            # Count how many times each model is best
            best_counts = {}
            for metric in metrics:
                best_model = self.get_best_model(metric)
                if best_model:
                    best_counts[best_model] = best_counts.get(best_model, 0) + 1

            most_consistent = max(best_counts.items(), key=lambda x: x[1])
            insights['consistency_analysis'] = {
                'most_consistent_model': most_consistent[0],
                'times_best': most_consistent[1],
                'total_metrics': len(metrics)
            }

        # Generate recommendations
        if len(success_df) > 1:
            # Recommend the most consistent model
            if insights['consistency_analysis']:
                consistent_model = insights['consistency_analysis']['most_consistent_model']
                insights['recommendations'].append(
                    f"Consider using '{consistent_model}' as it performs best across multiple metrics."
                )

            # Check for overfitting concerns (high CV variance)
            cv_std_cols = [col for col in success_df.columns if col.endswith('_cv_std')]
            if cv_std_cols:
                for col in cv_std_cols:
                    metric_name = col.replace('_cv_std', '')
                    high_variance = success_df[col] > success_df[col].median() * 1.5
                    if high_variance.any():
                        problematic_models = success_df[high_variance]['Model'].tolist()
                        insights['recommendations'].append(
                            f"Models {problematic_models} show high variance in {metric_name} - consider regularization."
                        )

        return insights
