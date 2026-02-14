from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional


@dataclass
class Contexto:
    gerador_reviews_steam: Optional[chain[Any]] = None
