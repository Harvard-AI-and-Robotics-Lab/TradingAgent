from typing import Dict, Optional
import os
import re

default_data_dir = os.path.join(
    os.path.dirname(__file__), "data_cache"
)

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = os.getenv("DATA_DIR", default_data_dir)
DATA_DIR_PREFIX: Optional[str] = {
    "simfin_balance_sheet": "fundamental_data/simfin_data_all/balance_sheet/companies/us",
    "simfin_cashflow": "fundamental_data/simfin_data_all/cash_flow/companies/us",
    "simfin_income_statements": "fundamental_data/simfin_data_all/income_statements/companies/us",
    "market_data": "market_data/price_data",

}
DATA_CACHE_FILE_YFIN: Optional[str] = os.getenv(
    "DATA_CACHE_FILE_YFIN",
    "YFin-data-2010-09-19-2025-09-19.csv"
)

# assert data file format as start date - end date
pattern = r"^YFin-data-\d{4}-\d{2}-\d{2}-\d{4}-\d{2}-\d{2}\.csv$"
assert re.match(pattern, DATA_CACHE_FILE_YFIN), (
    f"Invalid file name format: {DATA_CACHE_FILE_YFIN}"
)


def initialize_config():
    """Initialize the configuration with default values."""
    global _config, DATA_DIR, DATA_CACHE_FILE_YFIN
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
        if len(_config["data_dir"]) > 0:
            DATA_DIR = _config["data_dir"]
        if DATA_DIR is None or len(DATA_DIR) == 0:
            raise Exception(
                "DATA_DIR is not set {}".format(DATA_DIR)
            )

def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config, DATA_DIR, DATA_CACHE_FILE_YFIN
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)
    if len(_config["data_dir"]) > 0:
        DATA_DIR = _config["data_dir"]
    if DATA_DIR is None or len(DATA_DIR) == 0:
        raise Exception(
            "DATA_DIR is not set {}".format(DATA_DIR)
        )


def get_config() -> Dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


# Initialize with default config
initialize_config()
