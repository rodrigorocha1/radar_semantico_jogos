import itertools
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class Contexto(NamedTuple):
    gerador_reviews_steam: itertools.chain = None
