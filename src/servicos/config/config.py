import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()


class Config:
    STEAM_API_URL: Final[str] = os.getenv("STEAM_API_URL")
