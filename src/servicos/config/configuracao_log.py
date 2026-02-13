import logging
from pathlib import Path

log_file = Path(__file__).parent / "steam_api.log"

logger = logging.getLogger("steam_api_checker")
logger.setLevel(logging.INFO)

# Handler para arquivo
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)

# Handler para terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)

# Adiciona handlers ao logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
