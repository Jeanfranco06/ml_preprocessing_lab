"""
Helper Functions
Common utility functions for the application.
"""

import os
import re
import time
import functools
from pathlib import Path
from typing import Any, Callable, Union
import pandas as pd

def format_number(num: Union[int, float], decimals: int = 2) -> str:
    """
    Format a number with specified decimal places.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    if isinstance(num, (int, float)):
        if num == int(num):
            return str(int(num))
        return f"{num:.{decimals}f}"
    return str(num)

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as percentage.

    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"

def get_file_size(file_path: Union[str, Path]) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        Human-readable file size string
    """
    path = Path(file_path)
    if not path.exists():
        return "0 B"

    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def create_directory(dir_path: Union[str, Path]) -> bool:
    """
    Create directory if it doesn't exist.

    Args:
        dir_path: Directory path to create

    Returns:
        True if directory was created or already exists, False on error
    """
    try:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Safe filename
    """
    # Replace invalid characters with underscores
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing whitespace and dots
    safe_name = safe_name.strip(' .')
    # Ensure it's not empty
    if not safe_name:
        safe_name = "unnamed_file"
    return safe_name

def timer_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Human-readable memory usage string
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    return get_file_size_from_bytes(memory_bytes)

def get_file_size_from_bytes(size_bytes: int) -> str:
    """
    Convert bytes to human-readable file size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    size = size_bytes
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> dict:
    """
    Validate DataFrame structure.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Validation results dictionary
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }

    # Check if DataFrame is empty
    if df.empty:
        results['valid'] = False
        results['errors'].append("DataFrame is empty")
        return results

    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_cols}")

    # Check for duplicate column names
    if df.columns.duplicated().any():
        results['valid'] = False
        results['errors'].append("DataFrame has duplicate column names")

    # Check for completely null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        results['warnings'].append(f"Completely null columns: {null_columns}")

    # Info
    results['info'] = {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'memory_usage': get_memory_usage(df)
    }

    return results

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by removing special characters and standardizing.

    Args:
        df: DataFrame with columns to clean

    Returns:
        DataFrame with cleaned column names
    """
    df_clean = df.copy()

    def clean_name(name):
        # Convert to string if not already
        name = str(name)
        # Replace spaces with underscores
        name = name.replace(' ', '_')
        # Remove special characters except underscores
        name = re.sub(r'[^\w_]', '', name)
        # Convert to lowercase
        name = name.lower()
        # Remove multiple underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name if name else 'unnamed_column'

    df_clean.columns = [clean_name(col) for col in df_clean.columns]
    return df_clean

def sample_dataframe(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Get a sample of DataFrame rows.

    Args:
        df: DataFrame to sample
        n: Number of rows to sample
        random_state: Random state for reproducibility

    Returns:
        Sampled DataFrame
    """
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state).reset_index(drop=True)

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """
    Get comprehensive information about a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with DataFrame information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage': get_memory_usage(df),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }

    # Add numeric statistics if available
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        info['numeric_stats'] = df[numeric_cols].describe().to_dict()

    return info
