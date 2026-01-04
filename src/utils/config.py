"""
Configuration Management

Handles loading and merging configuration files.

Usage:
    from src.utils.config import load_config
    
    config = load_config('configs/default.yaml')
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class KeyboardConfig:
    """Keyboard detection configuration."""
    num_keys: int = 88
    num_white_keys: int = 52
    aspect_ratio: float = 8.147
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 100
    min_line_length: int = 100


@dataclass
class HandConfig:
    """Hand processing configuration."""
    num_landmarks: int = 21
    fingertip_indices: list = field(default_factory=lambda: [4, 8, 12, 16, 20])
    hampel_window: int = 20
    hampel_threshold: float = 3.0
    interpolation_max_gap: int = 30
    savgol_window: int = 11
    savgol_order: int = 3


@dataclass
class AssignmentConfig:
    """Finger assignment configuration."""
    sigma: float = 15.0
    candidate_keys: int = 2
    hand_separation_threshold: float = 0.5


@dataclass
class RefinementConfig:
    """Neural refinement configuration."""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 10


@dataclass
class Config:
    """Main configuration container."""
    project_name: str = "piano-fingering-detection"
    version: str = "1.0.0"
    
    # Data settings
    dataset_name: str = "PianoVAM/PianoVAM_v1.0"
    cache_dir: str = "./data/cache"
    video_fps: int = 60
    audio_sample_rate: int = 44100
    
    # Sub-configurations
    keyboard: KeyboardConfig = field(default_factory=KeyboardConfig)
    hand: HandConfig = field(default_factory=HandConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    
    # Evaluation
    metrics: list = field(default_factory=lambda: ["accuracy", "m_gen", "m_high", "ifr"])
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        config = cls()
        
        # Project settings
        project = config_dict.get('project', {})
        config.project_name = project.get('name', config.project_name)
        config.version = project.get('version', config.version)
        
        # Data settings
        data = config_dict.get('data', {})
        config.dataset_name = data.get('dataset_name', config.dataset_name)
        config.cache_dir = data.get('cache_dir', config.cache_dir)
        config.video_fps = data.get('video_fps', config.video_fps)
        config.audio_sample_rate = data.get('audio_sample_rate', config.audio_sample_rate)
        
        # Keyboard config
        keyboard = config_dict.get('keyboard', {})
        detection = keyboard.get('detection', {})
        config.keyboard = KeyboardConfig(
            num_keys=keyboard.get('num_keys', 88),
            num_white_keys=keyboard.get('num_white_keys', 52),
            aspect_ratio=keyboard.get('aspect_ratio', 8.147),
            canny_low=detection.get('canny_low', 50),
            canny_high=detection.get('canny_high', 150),
            hough_threshold=detection.get('hough_threshold', 100),
            min_line_length=detection.get('min_line_length', 100)
        )
        
        # Hand config
        hand = config_dict.get('hand', {})
        filtering = hand.get('filtering', {})
        config.hand = HandConfig(
            num_landmarks=hand.get('num_landmarks', 21),
            fingertip_indices=hand.get('fingertip_indices', [4, 8, 12, 16, 20]),
            hampel_window=filtering.get('hampel_window', 20),
            hampel_threshold=filtering.get('hampel_threshold', 3.0),
            interpolation_max_gap=filtering.get('interpolation_max_gap', 30),
            savgol_window=filtering.get('savgol_window', 11),
            savgol_order=filtering.get('savgol_order', 3)
        )
        
        # Assignment config
        assignment = config_dict.get('assignment', {})
        config.assignment = AssignmentConfig(
            sigma=assignment.get('sigma', 15.0),
            candidate_keys=assignment.get('candidate_keys', 2),
            hand_separation_threshold=assignment.get('hand_separation_threshold', 0.5)
        )
        
        # Refinement config
        refinement = config_dict.get('refinement', {})
        model = refinement.get('model', {})
        training = refinement.get('training', {})
        config.refinement = RefinementConfig(
            hidden_size=model.get('hidden_size', 128),
            num_layers=model.get('num_layers', 2),
            dropout=model.get('dropout', 0.3),
            bidirectional=model.get('bidirectional', True),
            batch_size=training.get('batch_size', 32),
            learning_rate=training.get('learning_rate', 0.001),
            epochs=training.get('epochs', 50),
            early_stopping_patience=training.get('early_stopping_patience', 10)
        )
        
        # Evaluation
        evaluation = config_dict.get('evaluation', {})
        config.metrics = evaluation.get('metrics', config.metrics)
        
        return config


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config.from_dict(config_dict)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Merge two config dictionaries.
    
    Args:
        base: Base configuration
        override: Override values
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Config, path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'project': {
            'name': config.project_name,
            'version': config.version
        },
        'data': {
            'dataset_name': config.dataset_name,
            'cache_dir': config.cache_dir,
            'video_fps': config.video_fps,
            'audio_sample_rate': config.audio_sample_rate
        },
        'keyboard': {
            'num_keys': config.keyboard.num_keys,
            'num_white_keys': config.keyboard.num_white_keys,
            'detection': {
                'canny_low': config.keyboard.canny_low,
                'canny_high': config.keyboard.canny_high,
                'hough_threshold': config.keyboard.hough_threshold
            }
        },
        'hand': {
            'num_landmarks': config.hand.num_landmarks,
            'fingertip_indices': config.hand.fingertip_indices,
            'filtering': {
                'hampel_window': config.hand.hampel_window,
                'hampel_threshold': config.hand.hampel_threshold,
                'interpolation_max_gap': config.hand.interpolation_max_gap,
                'savgol_window': config.hand.savgol_window,
                'savgol_order': config.hand.savgol_order
            }
        },
        'assignment': {
            'sigma': config.assignment.sigma,
            'candidate_keys': config.assignment.candidate_keys,
            'hand_separation_threshold': config.assignment.hand_separation_threshold
        },
        'refinement': {
            'model': {
                'hidden_size': config.refinement.hidden_size,
                'num_layers': config.refinement.num_layers,
                'dropout': config.refinement.dropout,
                'bidirectional': config.refinement.bidirectional
            },
            'training': {
                'batch_size': config.refinement.batch_size,
                'learning_rate': config.refinement.learning_rate,
                'epochs': config.refinement.epochs,
                'early_stopping_patience': config.refinement.early_stopping_patience
            }
        },
        'evaluation': {
            'metrics': config.metrics
        }
    }
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

