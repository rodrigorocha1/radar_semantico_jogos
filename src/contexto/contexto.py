import itertools
from typing import NamedTuple, Optional, Any


class Contexto(NamedTuple):
    gerador_reviews_steam: Optional[itertools.chain[Any]] = None
