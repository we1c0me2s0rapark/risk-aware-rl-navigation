import os
import sys
import yaml

try:
    from .logger import Log
except ImportError as e:
    print(f"[ERROR at {os.path.basename(__file__)}] {e}")

def load_config():
    """
    @brief Load the YAML configuration file from the module directory.

    Resolves the path of the current module, locates the associated
    configuration file (`config.yaml`), and safely parses its contents.

    @return dict | None Parsed configuration dictionary if successful;
            otherwise None if an error occurs.

    @note The configuration file must reside in the same directory as
          this module.

    @warning Returns None on failure. Callers should explicitly validate
             the result before use to avoid runtime errors.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    except Exception as e:
        Log.error(__file__, e)
        return None