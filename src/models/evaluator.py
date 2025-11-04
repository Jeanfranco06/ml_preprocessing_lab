"""
Model Evaluator
Handles evaluation and comparison of trained models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..config import get_config
from ..utils import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Evaluates trained machine learning models.
    """

    def __init__(self):
        """Initialize model evaluator."""
        self.evaluation_results = {}

    def evaluate_classification_model(self, model: Any, X_test: pd.DataFrame,
                                    y_test: pd.Series, model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a classification model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating classification model: {model_name}")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Get evaluation metrics from config
            eval_config = get_config("evaluation", {})
            metrics = eval_config.get("classification_metrics", ["accuracy", "precision", "recall", "f1_score"])

            results = {
                'model_name': model_name,
                'problem_type': 'classification',
                'metrics': {},
                'predictions': y_pred.tolist(),
                'true_values': y_test.tolist()
            }

            # Calculate metrics
            for metric in metrics:
                try:
                    if metric == "accuracy":
                        results['metrics'][metric] = accuracy_score(y_test, y_pred)
                    elif metric == "precision":
                        results['metrics'][metric] = precision_score(y_test, y_pred, average='weighted')
                    elif metric == "recall":
                        results['metrics'][metric] = recall_score(y_test, y_pred, average='weighted')
                    elif metric == "f1_score":
                        results['metrics'][metric] = f1_score(y_test, y_pred, average='weighted')
                    elif metric == "roc_auc":
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_test)
                            if len(np.unique(y_test)) == 2:  # Binary classification
                                results['metrics'][metric] = roc_auc_score(y_test, y_proba[:, 1])
                            else:  # Multi-class
                                results['metrics'][metric] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                        else:
                            results['metrics'][metric] = None
                except Exception as e:
                    logger.warning(f"Could not compute {metric}: {e}")
                    results['metrics'][metric] = None

            # Confusion matrix
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

            # Classification report
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)

            # Probability predictions if available
            if hasattr(model, 'predict_proba'):
                results['probabilities'] = model.predict_proba(X_test).tolist()

            self.evaluation_results[model_name] = results
            logger.info(f"Successfully evaluated model: {model_name}")

            return results

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'problem_type': 'classification'
            }

    def evaluate_regression_model(self, model: Any, X_test: pd.DataFrame,
                                y_test: pd.Series, model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a regression model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating regression model: {model_name}")

        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Get evaluation metrics from config
            eval_config = get_config("evaluation", {})
            metrics = eval_config.get("regression_metrics", ["mae", "mse", "rmse", "r2_score"])

            results = {
                'model_name': model_name,
                'problem_type': 'regression',
                'metrics': {},
                'predictions': y_pred.tolist(),
                'true_values': y_test.tolist()
            }

            # Calculate metrics
            for metric in metrics:
                try:
                    if metric == "mae":
                        results['metrics'][metric] = mean_absolute_error(y_test, y_pred)
                    elif metric == "mse":
                        results['metrics'][metric] = mean_squared_error(y_test, y_pred)
                    elif metric == "rmse":
                        results['metrics'][metric] = np.sqrt(mean_squared_error(y_test, y_pred))
                    elif metric == "r2_score":
                        results['metrics'][metric] = r2_score(y_test, y_pred)
                except Exception as e:
                    logger.warning(f"Could not compute {metric}: {e}")
                    results['metrics'][metric] = None

            # Residuals
            results['residuals'] = (y_test - y_pred).tolist()

            self.evaluation_results[model_name] = results
            logger.info(f"Successfully evaluated model: {model_name}")

            return results

        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {e}")
            return {
                'model_name': model_name,
                'error': str(e),
                'problem_type': 'regression'
            }

    def evaluate_models(self, models_dict: Dict[str, Any], X_test: pd.DataFrame,
                       y_test: pd.Series, problem_type: str = "classification") -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple models.

        Args:
            models_dict: Dictionary of model_name -> model_instance
            X_test: Test features
            y_test: Test target
            problem_type: Type of problem ("classification" or "regression")

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating {len(models_dict)} {problem_type} models")

        results = {}

        for model_name, model in models_dict.items():
            if model is None:
                continue

            if problem_type == "classification":
                result = self.evaluate_classification_model(model, X_test, y_test, model_name)
            elif problem_type == "regression":
                result = self.evaluate_regression_model(model, X_test, y_test, model_name)
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")

            results[model_name] = result

        logger.info(f"Evaluated {len(results)} models successfully")
        return results

    def get_evaluation_results(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get evaluation results.

        Args:
            model_name: Specific model name, or None for all results

        Returns:
            Evaluation results
        """
        if model_name:
            return self.evaluation_results.get(model_name, {})
        return self.evaluation_results

    def create_comparison_table(self, results: Dict[str, Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Create a comparison table of model performances.

        Args:
            results: Evaluation results dictionary, or None to use stored results

        Returns:
            Comparison DataFrame
        """
        if results is None:
            results = self.evaluation_results

        comparison_data = []

        for model_name, result in results.items():
            if 'error' in result:
                continue

            row = {'Model': model_name}

            # Add metrics
            if 'metrics' in result:
                for metric, value in result['metrics'].items():
                    if value is not None:
                        row[metric] = round(value, 4) if isinstance(value, (int, float)) else value

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def plot_confusion_matrix(self, model_name: str, figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix for a classification model.

        Args:
            model_name: Name of the evaluated model
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not evaluated")

        result = self.evaluation_results[model_name]
        if result.get('problem_type') != 'classification':
            raise ValueError(f"Model '{model_name}' is not a classification model")

        if 'confusion_matrix' not in result:
            raise ValueError(f"No confusion matrix available for model '{model_name}'")

        cm = np.array(result['confusion_matrix'])

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

        return fig

    def plot_residuals(self, model_name: str, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot residuals for a regression model.

        Args:
            model_name: Name of the evaluated model
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model '{model_name}' not evaluated")

        result = self.evaluation_results[model_name]
        if result.get('problem_type') != 'regression':
            raise ValueError(f"Model '{model_name}' is not a regression model")

        if 'residuals' not in result or 'predictions' not in result:
            raise ValueError(f"No residuals available for model '{model_name}'")

        residuals = np.array(result['residuals'])
        predictions = np.array(result['predictions'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predictions
        ax1.scatter(predictions, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predictions')
        ax1.grid(True, alpha=0.3)

        # Residuals distribution
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def get_best_model(self, metric: str = "accuracy", results: Dict[str, Dict[str, Any]] = None) -> str:
        """
        Get the name of the best performing model.

        Args:
            metric: Metric to use for comparison
            results: Evaluation results, or None to use stored results

        Returns:
            Name of the best model
        """
        if results is None:
            results = self.evaluation_results

        best_model = None
        best_score = float('-inf')

        for model_name, result in results.items():
            if 'error' in result or 'metrics' not in result:
                continue

            score = result['metrics'].get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model = model_name

        return best_model

    def generate_evaluation_report(self, model_name: str) -> str:
        """
        Generate a detailed evaluation report for a model.

        Args:
            model_name: Name of the model

        Returns:
            Markdown report string
        """
        if model_name not in self.evaluation_results:
            return f"# Model Evaluation Report\n\nError: Model '{model_name}' not found"

        result = self.evaluation_results[model_name]

        report = f"# Model Evaluation Report: {model_name}\n\n"
        report += f"**Problem Type:** {result.get('problem_type', 'Unknown')}\n\n"

        if 'error' in result:
            report += f"**Error:** {result['error']}\n"
            return report

        # Metrics
        if 'metrics' in result:
            report += "## Performance Metrics\n\n"
            report += "| Metric | Value |\n"
            report += "|--------|-------|\n"

            for metric, value in result['metrics'].items():
                if value is not None:
                    if isinstance(value, float):
                        report += f"| {metric} | {value:.4f} |\n"
                    else:
                        report += f"| {metric} | {value} |\n"
                else:
                    report += f"| {metric} | N/A |\n"

            report += "\n"

        # Classification report
        if 'classification_report' in result:
            report += "## Detailed Classification Report\n\n"
            report += "```\n"
            report += classification_report(
                result['true_values'],
                result['predictions'],
                target_names=None
            )
            report += "```\n\n"

        return report
