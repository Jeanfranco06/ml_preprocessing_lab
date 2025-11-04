"""
Model Trainer
Handles training of machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import importlib
from typing import Dict, List, Any, Optional, Tuple
import logging

from ..config import get_config
from ..utils import get_logger

logger = get_logger(__name__)

class ModelTrainer:
    """
    Handles training and evaluation of machine learning models.
    """

    def __init__(self, model_configs: List[Dict[str, Any]] = None):
        """
        Initialize model trainer.

        Args:
            model_configs: List of model configurations. If None, uses config.
        """
        if model_configs is None:
            # Determine model type based on available configs
            self.classification_models = get_config("models.classification", [])
            self.regression_models = get_config("models.regression", [])
        else:
            # Assume classification models if not specified
            self.classification_models = model_configs
            self.regression_models = []

        self.trained_models = {}
        self.cv_results = {}

    def _get_sklearn_model(self, model_config: Dict[str, Any]) -> Any:
        """
        Get sklearn model instance from configuration.

        Args:
            model_config: Model configuration dictionary

        Returns:
            Sklearn model instance
        """
        try:
            module_name = f"sklearn.{model_config.get('sklearn_module', 'linear_model')}"
            class_name = model_config['sklearn_name']

            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)

            params = model_config.get('params', {})
            return model_class(**params)

        except Exception as e:
            logger.error(f"Error creating model {model_config.get('name', 'Unknown')}: {e}")
            raise

    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          model_config: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """
        Train a single model.

        Args:
            X_train: Training features
            y_train: Training target
            model_config: Model configuration
            model_name: Name for the model

        Returns:
            Training results dictionary
        """
        if model_name is None:
            model_name = model_config.get('name', 'Unknown Model')

        logger.info(f"Training model: {model_name}")

        try:
            # Create model
            model = self._get_sklearn_model(model_config)

            # Train model
            model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = self._perform_cross_validation(model, X_train, y_train)

            # Store trained model
            self.trained_models[model_name] = model

            result = {
                'model': model,
                'name': model_name,
                'config': model_config,
                'cv_scores': cv_scores,
                'training_success': True
            }

            logger.info(f"Successfully trained model: {model_name}")
            return result

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            return {
                'model': None,
                'name': model_name,
                'config': model_config,
                'error': str(e),
                'training_success': False
            }

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        problem_type: str = "classification") -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models.

        Args:
            X_train: Training features
            y_train: Training target
            problem_type: Type of problem ("classification" or "regression")

        Returns:
            Dictionary of training results
        """
        logger.info(f"Training all {problem_type} models")

        if problem_type == "classification":
            model_configs = self.classification_models
        elif problem_type == "regression":
            model_configs = self.regression_models
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

        results = {}

        for model_config in model_configs:
            model_name = model_config.get('name', 'Unknown')
            result = self.train_single_model(X_train, y_train, model_config, model_name)
            results[model_name] = result

        successful_models = sum(1 for r in results.values() if r['training_success'])
        logger.info(f"Trained {successful_models}/{len(results)} models successfully")

        return results

    def _perform_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Args:
            model: Trained model
            X: Features
            y: Target
            cv_folds: Number of CV folds

        Returns:
            Cross-validation results
        """
        try:
            # Use stratified K-fold for classification
            if len(np.unique(y)) <= cv_folds:
                cv = StratifiedKFold(n_splits=len(np.unique(y)), shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Get scoring metrics from config
            eval_config = get_config("evaluation", {})
            if hasattr(model, "predict_proba"):  # Classification
                scoring = eval_config.get("classification_metrics", ["accuracy"])
            else:  # Regression
                scoring = eval_config.get("regression_metrics", ["neg_mean_squared_error"])

            cv_results = {}
            for metric in scoring:
                try:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                    cv_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                except Exception as e:
                    logger.warning(f"Could not compute {metric}: {e}")
                    cv_results[metric] = {'error': str(e)}

            return cv_results

        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            return {'error': str(e)}

    def get_trained_model(self, model_name: str) -> Any:
        """
        Get a trained model by name.

        Args:
            model_name: Name of the trained model

        Returns:
            Trained model instance
        """
        return self.trained_models.get(model_name)

    def get_available_models(self) -> List[str]:
        """
        Get list of available trained models.

        Returns:
            List of model names
        """
        return list(self.trained_models.keys())

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model to use
            X: Input features

        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.get_available_models()}")

        model = self.trained_models[model_name]
        return model.predict(X)

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make probability predictions using a trained model.

        Args:
            model_name: Name of the model to use
            X: Input features

        Returns:
            Probability predictions array or None if not available
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {self.get_available_models()}")

        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        return None

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a trained model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        if model_name not in self.trained_models:
            return {'error': f"Model '{model_name}' not found"}

        model = self.trained_models[model_name]

        info = {
            'name': model_name,
            'type': type(model).__name__,
            'module': type(model).__module__,
            'parameters': model.get_params() if hasattr(model, 'get_params') else {}
        }

        # Add feature importances if available
        if hasattr(model, 'feature_importances_'):
            info['feature_importances'] = model.feature_importances_.tolist()

        # Add coefficients if available
        if hasattr(model, 'coef_'):
            info['coefficients'] = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else model.coef_

        return info
