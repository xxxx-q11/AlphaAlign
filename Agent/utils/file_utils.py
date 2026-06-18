"""File operation utility class"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

import numpy as np
import pandas as pd


class FileUtils:
    """File operation utility collection"""
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Any:
        """
        Load JSON file
        
        Args:
            file_path: JSON file path
            
        Returns:
            Parsed JSON data
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file does not exist: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> str:
        """
        Save data to JSON file
        
        Args:
            data: Data to save
            file_path: Save path
            indent: Number of indentation spaces
            
        Returns:
            Saved file path
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        return str(file_path)
    
    @staticmethod
    def read_pickle_stats(file_path: Union[str, Path]) -> Optional[Dict[str, float]]:
        """
        Read pickle file and return statistics
        
        Args:
            file_path: Pickle file path
            
        Returns:
            Dictionary containing statistics, keys include:
            - count: Number of data points
            - mean: Mean value
            - std: Standard deviation
            - min/max: Minimum/maximum values
            - 25%/50%/75%: Quantiles
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File does not exist: {file_path}")
            return None
        
        if not file_path.is_file():
            print(f"Error: Path is not a file: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert to Series uniformly for statistics
            if isinstance(data, pd.Series):
                series = data
            elif isinstance(data, pd.DataFrame):
                if data.empty:
                    print("Warning: DataFrame is empty")
                    return None
                series = data.iloc[:, 0]
            elif isinstance(data, np.ndarray):
                if data.size == 0:
                    print("Warning: Array is empty")
                    return None
                series = pd.Series(data.flatten())
            else:
                print(f"Warning: Unsupported data type: {type(data).__name__}")
                return None
            
            stats = series.describe()
            return {
                'count': float(stats.get('count', 0)),
                'mean': float(stats.get('mean', 0)),
                'std': float(stats.get('std', 0)),
                'min': float(stats.get('min', 0)),
                '25%': float(stats.get('25%', 0)),
                '50%': float(stats.get('50%', 0)),
                '75%': float(stats.get('75%', 0)),
                'max': float(stats.get('max', 0))
            }
            
        except Exception as e:
            print(f"Error: Unable to read pickle file: {e}")
            return None
    
    @staticmethod
    def read_mlflow_metric(file_path: Union[str, Path]) -> Optional[float]:
        """
        Read MLflow metric file, return latest metric value
        
        Args:
            file_path: Metric file path
            
        Returns:
            Latest metric value, returns None if file is empty or does not exist
        """
        file_path = Path(file_path)
        #print(f"file_path: {file_path}")
        if not file_path.exists():
            print(f"Metric file does not exist: {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    #print(f"parts: {parts}")
                    if len(parts) >= 2:
                        return float(parts[1])
            print("Read failed")
            return None
        except Exception as e:
            print(f"Failed to read MLflow metric: {e}")
            return None
    
    @staticmethod
    def generate_hash_filename(content: str, prefix: str = "", suffix: str = "") -> str:
        """
        Generate filename with hash based on content
        
        Args:
            content: Content used to generate hash
            prefix: Filename prefix
            suffix: Filename suffix (e.g., .json)
            
        Returns:
            Generated filename
        """
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}{content_hash}_{timestamp}{suffix}"
    
    @staticmethod
    def ensure_dir(dir_path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, create if it doesn't exist
        
        Args:
            dir_path: Directory path
            
        Returns:
            Path object of the directory
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path


class ConfigLoader:
    """Configuration loader"""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def load_qlib_operators(cls) -> Dict[str, Any]:
        """
        Load Qlib operator configuration
        
        Returns:
            Operator configuration dictionary
        """
        cache_key = "qlib_operators"
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        config_path = Path(__file__).parent.parent / "config" / "qlib_operators.json"
        config = FileUtils.load_json(config_path)
        cls._cache[cache_key] = config
        return config
    
    @classmethod
    def clear_cache(cls):
        """Clear configuration cache"""
        cls._cache.clear()

