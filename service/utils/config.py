import logging
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

# --- Load environment settings once at import ---


def _find_env_settings_yaml() -> Path:
    """Search upwards for env_settings.yaml from this file's location."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        candidate = parent / "env_settings.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find env_settings.yaml in project structure")


_env_settings_path = _find_env_settings_yaml()
with open(_env_settings_path, "r") as f:
    ENV_SETTINGS: Dict[str, Any] = yaml.safe_load(f)


def get_env_settings() -> Dict[str, Any]:
    """Get environment settings."""
    return ENV_SETTINGS


def get_setting(key: str, default: Any = None) -> Any:
    """Get a specific environment setting."""
    return ENV_SETTINGS.get(key, default)

# --- Logging utilities ---


_log_file_path: Optional[str] = None


def setup_file_logging(log_dir: str = "logs") -> str:
    """
    Set up file logging and return the log file path.
    Ensures a FileHandler is always attached to the root logger.
    """
    global _log_file_path
    if _log_file_path:
        return _log_file_path

    os.makedirs(log_dir, exist_ok=True)
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file_path = os.path.join(log_dir, f"run_{run_id}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')

    # Remove all existing handlers (prevents duplicate logs)
    while root_logger.handlers:
        root_logger.handlers.pop()

    # Add file handler
    file_handler = logging.FileHandler(_log_file_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Add stream handler (console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    return _log_file_path


def get_log_file_path() -> Optional[str]:
    """Get the current log file path if file logging is enabled."""
    return _log_file_path


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance for the given name.
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    return logging.getLogger(name)
