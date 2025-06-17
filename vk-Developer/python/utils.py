import time
import logging
import json
import yaml
import os

def timeit(func):
    """Декоратор для замера времени выполнения"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"{func.__name__} executed in {end-start:.2f} seconds")
        return result
    return wrapper

def load_config():
    """Загружает конфигурацию из YAML-файла"""
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.warning("Config file not found, using defaults")
        return {
            'frames_per_second': 2,
            'batch_size': 16,
            'similarity_threshold': 0.82,
            'min_intro_length': 3,
            'max_intro_length': 30,
            'cleanup': True
        }

def setup_logger():
    """Настраивает логгер"""
    logger = logging.getLogger('IntroDetector')
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

def seconds_to_time_str(total_seconds):
    """
    Преобразует секунды в строку формата ЧЧ:ММ:СС
    """
    total_seconds = int(round(total_seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"