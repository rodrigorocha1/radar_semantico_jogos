import itertools
from dataclasses import dataclass


@dataclass
class Contexto:
    gerador_reviews_steam: itertools.chain = None

