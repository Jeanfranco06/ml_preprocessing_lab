"""
Configuration Manager
Loads and manages application configuration from YAML files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages application configuration loading and access.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        if config_path is None:
            # Default config path relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "settings.yaml"

        self.config_path = Path(config_path)
        self._config = {}
        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from YAML file.
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._config = self._get_default_config()
                return

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}

            logger.info(f"Configuration loaded from {self.config_path}")

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.

        Returns:
            Default configuration dictionary.
        """
        return {
            "app": {
                "name": "ML Preprocessing Suite",
                "version": "2.0.0",
                "description": "Advanced ML preprocessing pipeline",
                "author": "ML Suite Team"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/ml_preprocessing.log"
            },
            "datasets": {},
            "preprocessing": {
                "missing_values": {"numerical_strategy": "mean", "categorical_strategy": "mode"},
                "encoding": {"method": "label_encoding"},
                "scaling": {"method": "standard_scaler"}
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset configuration dictionary
        """
        return self.get(f"datasets.{dataset_name}", {})

    def get_all_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configuration for all datasets.

        Returns:
            Dictionary of all dataset configurations
        """
        return self.get("datasets", {})

    def get_model_config(self, model_type: str = "classification") -> list:
        """
        Get model configurations for a specific type.

        Args:
            model_type: Type of models ("classification" or "regression")

        Returns:
            List of model configurations
        """
        return self.get(f"models.{model_type}", [])

    def get_preprocessing_config(self) -> Dict[str, Any]:
        """
        Get preprocessing configuration.

        Returns:
            Preprocessing configuration dictionary
        """
        return self.get("preprocessing", {})

    def get_ui_config(self) -> Dict[str, Any]:
        """
        Get UI configuration.

        Returns:
            UI configuration dictionary
        """
        return self.get("ui", {})

    def get_export_config(self) -> Dict[str, Any]:
        """
        Get export configuration.

        Returns:
            Export configuration dictionary
        """
        return self.get("export", {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        Get evaluation configuration.

        Returns:
            Evaluation configuration dictionary
        """
        return self.get("evaluation", {})

    def reload_config(self) -> None:
        """
        Reload configuration from file.
        """
        self._load_config()

    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.

        Args:
            config_path: Path to save configuration. If None, uses current path.
        """
        save_path = Path(config_path) if config_path else self.config_path

        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise

    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of configuration updates
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    update_nested_dict(d[k], v)
                else:
                    d[k] = v

        update_nested_dict(self._config, updates)
        logger.info("Configuration updated")

# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager instance.

    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value using global config manager.

    Args:
        key: Configuration key
        default: Default value

    Returns:
        Configuration value
    """
    return get_config_manager().get(key, default)
