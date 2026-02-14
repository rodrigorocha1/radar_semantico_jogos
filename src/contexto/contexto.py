from dataclasses import dataclass
from itertools import chain
from typing import Any, Optional


@dataclass
class Contexto:
    gerador_reviews_steam: Optional[chain[Any]] = None
    gerador_comentarios_youtube: Optional[chain[Any]] = None
    gerador_resposta_comentarios_youtube: Optional[chain[Any]] = None
